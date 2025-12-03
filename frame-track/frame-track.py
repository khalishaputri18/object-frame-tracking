import cv2
import argparse
import datetime
import time
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 1. Logic Class
# ==========================================
class GapCounter:
    def __init__(self, gap_seconds, fps):
        self.gap_frames = int(gap_seconds * fps)
        self.last_seen_frame = {} 
        self.event_counts = {}   

    def get_event_id(self, class_name, current_frame):
        # Initialize if first time seeing this class
        if class_name not in self.event_counts:
            self.event_counts[class_name] = 1
            self.last_seen_frame[class_name] = current_frame
            return 1

        last_frame = self.last_seen_frame[class_name]
        delta = current_frame - last_frame
        
        # Always update last seen to keep the window open
        self.last_seen_frame[class_name] = current_frame

        if delta > self.gap_frames:
            # Gap exceeded -> New Event
            self.event_counts[class_name] += 1
            return self.event_counts[class_name]
        else:
            # Within gap -> Same Event
            return self.event_counts[class_name]

# ==========================================
# 2. Helper Functions
# ==========================================
def format_timestamp(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def get_unique_filepath(directory, base_name, extension):
    counter = 0
    while True:
        suffix = f"_{counter}" if counter > 0 else ""
        filename = f"{base_name}{suffix}{extension}"
        file_path = directory / filename
        if not file_path.exists():
            return file_path
        counter += 1

def draw_counter_summary(frame, counts_dict, width):
    if not counts_dict:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    margin = 20
    line_spacing = 30
    start_y = 40
    
    sorted_keys = sorted(counts_dict.keys())

    for i, cls_name in enumerate(sorted_keys):
        count = counts_dict[cls_name]
        text = f"{cls_name.upper()}: {count}"
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x_pos = width - text_w - margin
        y_pos = start_y + (i * line_spacing)
        
        # Draw background and text
        cv2.rectangle(frame, (x_pos - 5, y_pos - text_h - 5), (width - margin + 5, y_pos + 5), bg_color, -1)
        cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, text_color, thickness)

# ==========================================
# 3. Single Video Processor
# ==========================================
def process_single_video(model, video_path, args, output_dir):
    print(f"\n--- Processing: {video_path.name} ---")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Video Info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    
    if fps == 0: return

    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS, Total Frames: {total_frames}")
    print(f"Confidence Threshold: {args.conf}")

    # Initialize Logic
    counter_logic = GapCounter(args.gap, fps)
    
    # Setup Outputs
    input_stem = video_path.stem 
    label_path = get_unique_filepath(output_dir, f"{input_stem}_labels", ".txt")
    
    video_out_path = None
    out_writer = None
    
    if not args.nosave:
        video_out_path = get_unique_filepath(output_dir, f"{input_stem}_output", ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
        print(f"Video Output: {video_out_path}")

    print(f"Label Output: {label_path}")

    # Open Label File
    with open(label_path, 'w') as txt_file:
        txt_file.write("Frame,Timestamp,Class,Event_ID,Conf,x1,y1,x2,y2\n") 
        
        frame_idx = 0
        
        # --- OPTIMIZATION VARS ---
        print_interval = 10  # Print log every 10 frames
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            timestamp_str = format_timestamp(frame_idx / fps)

            # 1. Inference
            # verbose=False prevents internal YOLO printing, speeding things up
            results = model.predict(frame, classes=args.classes, verbose=False, conf=args.conf)
            det_results = results[0]

            # 2. Process Detections
            if len(det_results.boxes) > 0:
                for box in det_results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    confidence = float(box.conf[0]) 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get Count ID
                    event_id = counter_logic.get_event_id(cls_name, frame_idx)

                    # Write to CSV
                    row = f"{frame_idx},{timestamp_str},{cls_name},{event_id},{confidence:.2f},{x1},{y1},{x2},{y2}\n"
                    txt_file.write(row)

                    # Draw Boxes (only if saving or showing)
                    if not args.nosave or args.show:
                        label = f"{cls_name} {confidence:.2f} ID:{event_id}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 3. Optimized Logging (Every 10 frames)
            if frame_idx % print_interval == 0:
                curr_time = time.time()
                elapsed_time = curr_time - prev_time
                
                if elapsed_time > 0:
                    current_fps = print_interval / elapsed_time
                else:
                    current_fps = 0.0
                
                progress_str = f"{frame_idx}/{total_frames}"
                print(f"[Frame {progress_str} | {timestamp_str}] Speed: {current_fps:.1f} FPS")
                
                # Reset timer
                prev_time = curr_time

            # 4. Final Visuals
            if not args.nosave or args.show:
                draw_counter_summary(frame, counter_logic.event_counts, width)

            if not args.nosave:
                out_writer.write(frame)

            if args.show:
                cv2.imshow(f'Processing {input_stem}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if out_writer:
        out_writer.release()
    if args.show:
        cv2.destroyWindow(f'Processing {input_stem}')

# ==========================================
# 4. Main Execution
# ==========================================
def run():
    parser = argparse.ArgumentParser(description="Batch Video Counter Optimized")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to model')
    parser.add_argument('--source', type=str, required=True, help='Path to mp4 file OR folder')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter class IDs')
    parser.add_argument('--gap', type=float, default=5.0, help='Time gap (seconds)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence Threshold (0.0 - 1.0)')
    parser.add_argument('--nosave', action='store_true', help='Do not save video')
    parser.add_argument('--show', action='store_true', help='Show preview window')
    
    args = parser.parse_args()

    source_path = Path(args.source)
    video_files = []

    # Handle File vs Folder
    if source_path.is_file() and source_path.suffix == '.mp4':
        video_files.append(source_path)
    elif source_path.is_dir():
        video_files = list(source_path.glob('*.mp4'))
    else:
        print(f"Error: Invalid source {args.source}")
        return

    if not video_files:
        print("No .mp4 files found.")
        return

    # Create Output Directory
    output_dir = Path("inference_output3")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading Model {args.model}...")
    model = YOLO(args.model)

    # Run Batch
    for video in video_files:
        process_single_video(model, video, args, output_dir)
    
    print("\nAll tasks completed.")

if __name__ == "__main__":
    run()