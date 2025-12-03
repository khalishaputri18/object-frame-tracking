import os
import cv2
import numpy as np
import time
import argparse
import glob
import sys
import datetime # <--- Added for Timestamp
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIGURATION ---
TRAIL_LENGTH = 0 

def process_video_deepsort(args):
    source = args['source']
    track_ = args['track'] 
    count_ = args['count']
    batch_mode = args['batch']
    target_classes = args['classes']
    model_path = args['model']    
    conf_threshold = args['conf']
    nosave = args['nosave']

    # Initialize Model
    print(f"Loading Model: {model_path} with Conf: {conf_threshold}")
    try:
        model = YOLO(model_path)
        names = model.names
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return

    tracker = DeepSort(max_age=15, n_init=5, nms_max_overlap=1.0)
    
    # --- PER-CLASS ID MAPPING VARIABLES ---
    trajectories = {}
    class_id_mappings = {}  # {class_id: {raw_deepsort_id: clean_id}}
    class_counters = {}     # {class_id: next_available_id}
    # --------------------------------------

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_input_name = os.path.splitext(os.path.basename(source))[0] if source != '0' else 'webcam'
    
    # --- FILE NAMING LOGIC ---
    base_name_no_ext = os.path.join(output_dir, f'{base_input_name}_deepsort')
    final_video_path = f"{base_name_no_ext}.mp4"
    # Changed extension to .csv
    final_text_path = f"{base_name_no_ext}.csv"

    counter = 0
    # Check existence to prevent overwrites
    while os.path.exists(final_video_path) or os.path.exists(final_text_path):
        counter += 1
        final_video_path = f"{base_name_no_ext}_{counter}.mp4"
        final_text_path = f"{base_name_no_ext}_{counter}.csv"

    print(f"[{base_input_name}] Processing: {source}")
    print(f"[{base_input_name}] Labels: {final_text_path}")
    
    # --- WRITE CSV HEADER ---
    with open(final_text_path, 'w') as f:
        f.write("Frame,Timestamp,Class,ID,Conf,x1,y1,x2,y2\n")

    if nosave:
        print(f"[{base_input_name}] Video: DISABLED (--nosave active)")
    else:
        print(f"[{base_input_name}] Video: {final_video_path}")

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30 if (fps == 0 or np.isnan(fps)) else int(fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize VideoWriter ONLY if not saving
    out = None
    if not nosave:
        try: fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        except: fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(final_video_path, fourcc, fps, (w, h))

    frameId = 0
    start_time = time.time()
    real_fps_str = "Init..."
    model_fps_str = "Model: N/A"
    object_count = 0
    
    np.random.seed(42)
    id_colors = np.random.randint(0, 255, size=(10000, 3), dtype='uint8')

    while True:
        ret, frame = cap.read()
        if not ret: break
        frameId += 1

        # --- TIMESTAMP CALCULATION ---
        current_seconds = frameId / fps
        timestamp_str = str(datetime.timedelta(seconds=int(current_seconds)))

        t1 = time.time()
        results = model.predict(frame, conf=conf_threshold, classes=target_classes, verbose=False)
        t2 = time.time()
        
        inference_ms = (t2 - t1) * 1000
        if inference_ms > 0: model_fps_str = f"Model FPS: {(1000.0 / inference_ms):.1f}"

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width = x2 - x1
                height = y2 - y1
                conf = float(box.conf)
                cls_id = int(box.cls)
                detections.append([[x1, y1, width, height], conf, cls_id])

        tracks = tracker.update_tracks(detections, frame=frame)
        active_tracks = [t for t in tracks if t.is_confirmed()]
        object_count = len(active_tracks)

        current_raw_ids = [t.track_id for t in active_tracks]
        
        # Cleanup Old Trajectories
        for key in list(trajectories.keys()):
            if key not in current_raw_ids:
                del trajectories[key]

        # --- WRITE LABELS & DRAW ---
        with open(final_text_path, 'a') as f:
            for track in active_tracks:
                raw_id = track.track_id
                cls_id = int(track.det_class)
                
                # --- NEW PER-CLASS ID MAPPING ---
                if cls_id not in class_counters:
                    class_counters[cls_id] = 1
                    class_id_mappings[cls_id] = {}
                
                if raw_id not in class_id_mappings[cls_id]:
                    class_id_mappings[cls_id][raw_id] = class_counters[cls_id]
                    class_counters[cls_id] += 1
                
                visual_id = class_id_mappings[cls_id][raw_id]
                # -------------------------------

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                conf = track.det_conf if track.det_conf is not None else 0.0
                class_name = names.get(cls_id, "Unknown")
                conf_str = f"{conf:.2f}"
                
                # --- WRITE CSV LINE ---
                # Format: Frame, Timestamp, Class, ID, Conf, x1, y1, x2, y2
                line = f"{frameId},{timestamp_str},{class_name},{visual_id},{conf_str},{x1},{y1},{x2},{y2}\n"
                f.write(line)

                # Draw
                label = f"{class_name}: {visual_id} ({conf_str})"
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                if raw_id not in trajectories:
                    trajectories[raw_id] = deque(maxlen=TRAIL_LENGTH)
                trajectories[raw_id].append((center_x, center_y))

                color_seed = visual_id + (cls_id * 50)
                color = [int(c) for c in id_colors[color_seed % len(id_colors)]]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if frameId % 10 == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            if elapsed > 0:
                real_fps = 10 / elapsed
                real_fps_str = f"Real FPS: {real_fps:.1f}"
            start_time = time.time()
            if batch_mode:
                print(f"[{base_input_name}] {frameId}/{total_frames} | {real_fps_str} | Count: {object_count}")

        # HUD / OSD
        if not nosave or not batch_mode:
            cv2.putText(frame, real_fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, model_fps_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            if count_:
                y_offset = 30
                num_classes = len(class_counters)
                box_h = 30 * (num_classes + 1) if num_classes > 0 else 40
                
                cv2.rectangle(frame, (w - 220, 0), (w, box_h), (0,0,0), -1)
                cv2.putText(frame, "Total Counts:", (w - 210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if num_classes == 0:
                     cv2.putText(frame, "0", (w - 210, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    for cls_id, count_val in class_counters.items():
                        y_offset += 30
                        actual_count = count_val - 1
                        cls_name = names.get(cls_id, str(cls_id))
                        if len(cls_name) > 10: cls_name = cls_name[:10]
                        
                        text = f"{cls_name}: {actual_count}"
                        cv2.putText(frame, text, (w - 210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if not nosave and out is not None:
            out.write(frame)
            
        if not batch_mode:
            cv2.imshow(f"DeepSORT_{source}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if out is not None: out.release()
    if not batch_mode: cv2.destroyAllWindows()
    print(f"[{base_input_name}] Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', nargs='+', type=str, default=['0'], help='Video file or Folder')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--track', action='store_true', default=True)
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int) 
    parser.add_argument('--nosave', action='store_true', help='Do not save output video')
    args = parser.parse_args()

    # --- FOLDER EXPANSION LOGIC ---
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
        'track': args.track, 
        'count': args.count, 
        'batch': args.batch, 
        'classes': args.classes, 
        'model': args.model, 
        'conf': args.conf,
        'nosave': args.nosave
    } for s in video_sources]

    for i, args_dict in enumerate(process_args_list):
        print(f"\n--- Video {i+1} of {len(process_args_list)} ---")
        process_video_deepsort(args_dict)
    
    print("\nAll processing complete.")