"""Producer-consumer parallel pipeline for video processing."""

import threading
import queue
from typing import Optional, Callable, List
from tqdm import tqdm

from .video import VideoReader
from .compression import AutoVideoWriter
from .diff_engine import DiffMode, compute_diff
from .gpu_backend import to_gpu, to_cpu, estimate_batch_size


class PipelineError(Exception):
    """Exception raised during pipeline execution."""
    pass


class VideoPipeline:
    """
    Multi-threaded video processing pipeline.
    
    Stages:
    1. Reader: Reads frames from disk
    2. Batcher: Groups frames into batches
    3. Compute: Processes diffs (GPU or CPU)
    4. Writer: Writes output frames
    """
    
    def __init__(self, reader: VideoReader, writer: AutoVideoWriter,
                 mode: DiffMode, threshold: int = 0,
                 batch_size: Optional[int] = None, use_gpu: bool = False):
        self.reader = reader
        self.writer = writer
        self.mode = mode
        self.threshold = threshold
        self.use_gpu = use_gpu
        
        # Auto-detect batch size if not specified
        if batch_size is None:
            frame_shape = (reader.metadata.height, reader.metadata.width, 3)
            self.batch_size = estimate_batch_size(frame_shape)
        else:
            self.batch_size = max(1, batch_size)
        
        print(f"Batch size: {self.batch_size}")
        
        # Queues for inter-stage communication (bounded to limit memory)
        self.frame_queue = queue.Queue(maxsize=8)
        self.batch_queue = queue.Queue(maxsize=4)
        self.output_queue = queue.Queue(maxsize=8)
        
        # Error handling
        self.error_event = threading.Event()
        self.error_message = None
        
        # Progress tracking
        self.total_frames = reader.metadata.total_frames - 1  # k-1 diffs
        self.processed_frames = 0
    
    def _set_error(self, message: str):
        """Set error state from worker thread."""
        if not self.error_event.is_set():
            self.error_message = message
            self.error_event.set()
    
    def _reader_thread(self):
        """Read all frames from video and push to frame queue."""
        try:
            for frame in self.reader.frames():
                if self.error_event.is_set():
                    break
                self.frame_queue.put(frame)
            # Signal end of frames
            self.frame_queue.put(None)
        except Exception as e:
            self._set_error(f"Reader error: {e}")
            self.frame_queue.put(None)
    
    def _batch_thread(self):
        """Group frames into consecutive pairs and batch them."""
        try:
            prev_frame = self.frame_queue.get()
            if prev_frame is None:
                self.batch_queue.put(None)
                return
            
            batch_prev = []
            batch_curr = []
            
            while True:
                if self.error_event.is_set():
                    break
                
                curr_frame = self.frame_queue.get()
                if curr_frame is None:
                    # Send remaining batch
                    if batch_prev:
                        self.batch_queue.put((batch_prev, batch_curr))
                    self.batch_queue.put(None)
                    break
                
                batch_prev.append(prev_frame)
                batch_curr.append(curr_frame)
                
                if len(batch_prev) >= self.batch_size:
                    self.batch_queue.put((batch_prev, batch_curr))
                    batch_prev = []
                    batch_curr = []
                
                prev_frame = curr_frame
                
        except Exception as e:
            self._set_error(f"Batch error: {e}")
            self.batch_queue.put(None)
    
    def _compute_thread(self):
        """Process batches and compute diff frames."""
        try:
            while True:
                if self.error_event.is_set():
                    break
                
                batch = self.batch_queue.get()
                if batch is None:
                    self.output_queue.put(None)
                    break
                
                batch_prev, batch_curr = batch
                
                # Process batch
                if self.use_gpu:
                    diff_frames = self._process_batch_gpu(batch_prev, batch_curr)
                else:
                    diff_frames = self._process_batch_cpu(batch_prev, batch_curr)
                
                for diff_frame in diff_frames:
                    self.output_queue.put(diff_frame)
                    
        except Exception as e:
            self._set_error(f"Compute error: {e}")
            self.output_queue.put(None)
    
    def _process_batch_cpu(self, batch_prev: List, batch_curr: List) -> List:
        """Process a batch on CPU."""
        results = []
        for prev, curr in zip(batch_prev, batch_curr):
            diff = compute_diff(prev, curr, self.mode, self.threshold)
            results.append(diff)
        return results
    
    def _process_batch_gpu(self, batch_prev: List, batch_curr: List) -> List:
        """Process a batch on GPU."""
        import numpy as np
        from .gpu_backend import get_array_module
        
        xp = get_array_module()
        
        # Stack into batch arrays
        prev_stack = np.stack(batch_prev)  # (B, H, W, 3)
        curr_stack = np.stack(batch_curr)  # (B, H, W, 3)
        
        # Transfer to GPU
        prev_gpu = to_gpu(prev_stack)
        curr_gpu = to_gpu(curr_stack)
        
        # Compute batch diff
        f1 = prev_gpu.astype(xp.int16)
        f2 = curr_gpu.astype(xp.int16)
        diff = xp.abs(f2 - f1)  # (B, H, W, 3)
        
        # Threshold mask
        changed_mask = xp.any(diff > self.threshold, axis=3)  # (B, H, W)
        
        # Process based on mode
        if self.mode == DiffMode.GRAYSCALE:
            avg_diff = xp.mean(diff, axis=3)  # (B, H, W)
            intensity = 255 - avg_diff
            intensity = xp.where(changed_mask, intensity, 0)
            intensity = xp.clip(intensity, 0, 255).astype(xp.uint8)
            result = xp.stack([intensity, intensity, intensity], axis=3)
        else:  # COLOR mode
            result = xp.where(changed_mask[:, :, :, xp.newaxis], curr_gpu, 0)
            result = result.astype(xp.uint8)
        
        # Transfer back to CPU
        result_cpu = to_cpu(result)
        
        # Split batch into individual frames
        return [result_cpu[i] for i in range(len(batch_prev))]
    
    def _writer_thread(self):
        """Write output frames to video."""
        try:
            while True:
                if self.error_event.is_set():
                    break
                
                frame = self.output_queue.get()
                if frame is None:
                    break
                
                self.writer.write(frame)
                self.processed_frames += 1
                
        except Exception as e:
            self._set_error(f"Writer error: {e}")
    
    def run(self) -> int:
        """
        Run the complete pipeline.
        
        Returns:
            Number of frames processed
        """
        # Create and start threads
        threads = [
            threading.Thread(target=self._reader_thread, name="Reader"),
            threading.Thread(target=self._batch_thread, name="Batcher"),
            threading.Thread(target=self._compute_thread, name="Compute"),
            threading.Thread(target=self._writer_thread, name="Writer"),
        ]
        
        for t in threads:
            t.start()
        
        # Progress bar
        with tqdm(total=self.total_frames, desc="Processing") as pbar:
            last_count = 0
            while any(t.is_alive() for t in threads):
                # Update progress
                current = self.processed_frames
                if current > last_count:
                    pbar.update(current - last_count)
                    last_count = current
                
                # Check for errors
                if self.error_event.is_set():
                    # Signal all queues to unblock
                    self.frame_queue.put(None)
                    self.batch_queue.put(None)
                    self.output_queue.put(None)
                    break
                
                # Short sleep to avoid busy waiting
                import time
                time.sleep(0.1)
        
        # Wait for all threads to finish
        for t in threads:
            t.join(timeout=5)
        
        # Check for errors
        if self.error_event.is_set():
            raise PipelineError(self.error_message)
        
        return self.processed_frames
