import os
import cv2
import numpy as np
import time
import argparse
import glob
import sys
import datetime  # <--- Added for Timestamp calculation
from collections import deque
from ultralytics import YOLO

# --- DEFAULTS ---
DEFAULT_MODEL = "../yolov8n_no-pretrained_258ep/weights/best.pt" 
CONF_THRESHOLD = 0.7          
MIN_HISTORY = 5                

def process_frame(image, results, labels_save_path, frame_idx, timestamp, state, target_classes, names, colors, id_colors, track=True):
    """
    Processes a single frame: handles tracking logic, PER-CLASS ID mapping, drawing, and CSV writing.
    """
    tracking_trajectories = state['trajectories']
    class_id_mappings = state['class_id_mappings'] 
    class_counters = state['class_counters']

    bboxes = []
    model_fps_str = "Model: N/A"

    if results:
        speed = results[0].speed
        total_ms = speed.get('preprocess', 0) + speed.get('inference', 0) + speed.get('postprocess', 0)
        if total_ms > 0:
            model_fps_str = f"Model FPS: {(1000.0 / total_ms):.1f}"

    # Open file in Append mode
    with open(labels_save_path, 'a') as file:
        
        # ---------------------------------------------------------
        # MODE 1: TRACKING
        # ---------------------------------------------------------
        if track:
            current_raw_ids = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                current_raw_ids = [int(x) for x in results[0].boxes.id.tolist()]
            
            # Cleanup trajectories
            for id_ in list(tracking_trajectories.keys()):
                if id_ not in current_raw_ids:
                    del tracking_trajectories[id_]

            for predictions in results:
                if predictions is None or predictions.boxes is None or predictions.boxes.id is None:
                    continue

                for bbox in predictions.boxes:
                    raw_id = int(bbox.id)
                    conf = float(bbox.conf)
                    cls = int(bbox.cls)
                    xyxy = bbox.xyxy[0].cpu().numpy()
                    xmin, ymin, xmax, ymax = map(int, xyxy)
                    
                    if conf < CONF_THRESHOLD: continue 

                    # Update Trajectory
                    centroid_x = (xmin + xmax) / 2
                    centroid_y = (ymin + ymax) / 2
                    
                    if raw_id not in tracking_trajectories:
                        tracking_trajectories[raw_id] = deque(maxlen=30)
                    tracking_trajectories[raw_id].append((centroid_x, centroid_y))
                    
                    if len(tracking_trajectories[raw_id]) < MIN_HISTORY: continue 

                    # --- PER-CLASS ID LOGIC ---
                    if cls not in class_id_mappings:
                        class_id_mappings[cls] = {}
                        class_counters[cls] = 1 

                    if raw_id not in class_id_mappings[cls]:
                        class_id_mappings[cls][raw_id] = class_counters[cls]
                        class_counters[cls] += 1
                    
                    clean_id = class_id_mappings[cls][raw_id]

                    # Drawing
                    color_seed = clean_id + (cls * 50) 
                    color_bgr = [int(c) for c in id_colors[color_seed % len(id_colors)]]
                    
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_bgr, 2)
                    
                    class_name = names.get(cls, str(cls))
                    label = f'{class_name}: {clean_id} ({int(conf * 100)}%)'
                    
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(image, (xmin, ymin - 20), (xmin + t_size[0], ymin), color_bgr, -1)
                    cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    bboxes.append([xyxy, conf, cls, clean_id])

        # ---------------------------------------------------------
        # MODE 2: PREDICT (No Tracking)
        # ---------------------------------------------------------
        else:
            for predictions in results:
                if predictions is None or predictions.boxes is None: continue
                
                for bbox in predictions.boxes:
                    xyxy = bbox.xyxy[0].cpu().numpy()
                    xmin, ymin, xmax, ymax = map(int, xyxy)
                    conf = float(bbox.conf)
                    cls = int(bbox.cls)

                    if conf < CONF_THRESHOLD: continue
                    
                    color_bgr = [int(c) for c in colors[cls % len(colors)]]
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_bgr, 2)
                    class_name = names.get(cls, str(cls))
                    label = f'{class_name} {int(conf * 100)}%'
                    cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    bboxes.append([xyxy, conf, cls, -1])

        # --- CSV WRITING ---
        for item in bboxes:
            bbox_coords, scores, classes, id_ = item
            cls_name = names.get(classes, str(classes))
            
            # Format: Frame, Timestamp, Class, ID, Conf, x1, y1, x2, y2
            # Note: We use the 'clean_id' (id_) calculated above
            line = f"{frame_idx},{timestamp},{cls_name},{id_},{scores:.2f},{int(bbox_coords[0])},{int(bbox_coords[1])},{int(bbox_coords[2])},{int(bbox_coords[3])}\n"
            file.write(line)

    return image, len(bboxes), model_fps_str


def process_video(args):
    source = args['source']
    track_ = args['track']
    count_ = args['count']
    batch_mode = args['batch']
    target_classes = args['classes']
    nosave = args['nosave']
    model_path = args['model']
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return

    names = model.names
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')
    id_colors = np.random.randint(0, 255, size=(10000, 3), dtype='uint8')

    state = {
        'trajectories': {},
        'class_id_mappings': {}, 
        'class_counters': {}     
    }

    # Setup Paths
    output_dir = 'output'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    base_input_name = os.path.splitext(os.path.basename(source))[0]
    base_video_path = os.path.join(output_dir, f'{base_input_name}_output.mp4')
    
    final_video_path = base_video_path
    unique_suffix = "" 
    counter = 0

    name_no_ext, ext = os.path.splitext(base_video_path)
    # Changed extension to .csv
    final_text_path = os.path.join(output_dir, f'{base_input_name}_labels.csv')
    
    while os.path.exists(final_video_path) or os.path.exists(final_text_path):
        counter += 1
        unique_suffix = f"_{counter}"
        final_video_path = f"{name_no_ext}{unique_suffix}{ext}"
        final_text_path = os.path.join(output_dir, f'{base_input_name}_labels{unique_suffix}.csv')

    print(f"[{base_input_name}] Processing...")
    print(f"[{base_input_name}] Labels: {final_text_path}")
    
    # --- WRITE CSV HEADER ---
    with open(final_text_path, 'w') as f:
        f.write("Frame,Timestamp,Class,ID,Conf,x1,y1,x2,y2\n")

    if nosave:
        print(f"[{base_input_name}] Video: DISABLED (--nosave active)")
    else:
        print(f"[{base_input_name}] Video: {final_video_path}")

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"Error: Could not open {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30 if (fps == 0 or np.isnan(fps)) else int(fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if not nosave:
        try: fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        except: fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(final_video_path, fourcc, fps, (frame_width, frame_height))

    frameId = 0
    start_time = time.time()
    real_fps_str = "Init..."
    model_fps_str = "Init..."
    object_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frameId += 1
        
        # --- TIMESTAMP CALCULATION ---
        current_seconds = frameId / fps
        timestamp_str = str(datetime.timedelta(seconds=int(current_seconds)))

        if track_:
            results = model.track(frame, verbose=False, device=0, persist=True, 
                                  tracker="bytetrack.yaml", classes=target_classes,
                                  conf=CONF_THRESHOLD)
        else:
            results = model.predict(frame, verbose=False, device=0, 
                                    classes=target_classes, conf=CONF_THRESHOLD)

        # Passed frameId and timestamp_str to process_frame
        frame, object_count, model_fps_str = process_frame(
            frame, results, final_text_path, frameId, timestamp_str, state, target_classes, names, colors, id_colors, track_
        )

        if frameId % 10 == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            if elapsed > 0:
                real_fps = 10 / elapsed
                real_fps_str = f"Real FPS: {real_fps:.1f}"
            start_time = time.time()
            if batch_mode:
                print(f"[{base_input_name}] {frameId}/{total_frames} | {real_fps_str} | Count: {object_count}")

        if not nosave or not batch_mode:
            cv2.putText(frame, real_fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, model_fps_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            if count_:
                y_offset = 30
                num_classes = len(state['class_counters'])
                box_h = 30 * (num_classes + 1) if num_classes > 0 else 40
                cv2.rectangle(frame, (frame_width - 220, 0), (frame_width, box_h), (0,0,0), -1)

                cv2.putText(frame, "Total Counts:", (frame_width - 210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if num_classes == 0:
                     cv2.putText(frame, "0", (frame_width - 210, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    for cls_id, count_val in state['class_counters'].items():
                        y_offset += 30
                        actual_count = count_val - 1
                        cls_name = names.get(cls_id, str(cls_id))
                        if len(cls_name) > 10: cls_name = cls_name[:10]
                        
                        text = f"{cls_name}: {actual_count}"
                        cv2.putText(frame, text, (frame_width - 210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if not nosave and out is not None:
            out.write(frame)

        if not batch_mode:
            cv2.imshow(f"yolo_{source}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out is not None: out.release()
    if not batch_mode: cv2.destroyAllWindows()
    print(f"[{base_input_name}] Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video with YOLO.')
    parser.add_argument('--source', nargs='+', type=str, default=['0'], help='Video file(s) or Folder path(s)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Path to YOLO .pt model')
    parser.add_argument('--track', action='store_true', help='Enable tracking')
    parser.add_argument('--count', action='store_true', help='Show object count on screen')
    parser.add_argument('--batch', action='store_true', help='Batch mode (no display window)')
    parser.add_argument('--classes', nargs='+', type=int, help='Class IDs to track (e.g., 0 2 3)')
    parser.add_argument('--nosave', action='store_true', help='Do not save output video file')
    
    args = parser.parse_args()

    video_sources = []
    
    for path in args.source:
        if os.path.isdir(path):
            print(f"Detected folder: {path}. Scanning for .mp4 files...")
            files = glob.glob(os.path.join(path, '**', '*.mp4'), recursive=True)
            video_sources.extend(files)
            print(f"  Found {len(files)} videos in folder.")
        elif os.path.isfile(path):
            video_sources.append(path)
        elif path == '0' or path.isdigit():
            video_sources.append(path)
        else:
            print(f"Warning: Skipping invalid path '{path}'")

    video_sources = list(set(video_sources))
    video_sources.sort() 

    if not video_sources:
        print("No valid video sources found!")
        sys.exit()

    print(f"Total videos to process: {len(video_sources)}")

    process_args_list = [{
        'source': s, 
        'model': args.model,
        'track': args.track, 
        'count': args.count, 
        'batch': args.batch,
        'classes': args.classes,
        'nosave': args.nosave
    } for s in video_sources]

    print("\nStarting sequential processing...")
    for i, args_dict in enumerate(process_args_list):
        print(f"\n--- Video {i+1} of {len(process_args_list)} ---")
        process_video(args_dict)
    
    print("\nAll videos processed.")