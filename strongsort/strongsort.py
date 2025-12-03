import os
import cv2
import numpy as np
import time
import argparse
import glob
import sys
import datetime # <--- Added for Timestamp
from collections import deque
from pathlib import Path
from ultralytics import YOLO

# --- IMPORT FIX ---
try:
    from boxmot.trackers.strongsort.strongsort import StrongSort
except ImportError:
    print("Could not import StrongSort. Check your boxmot version.")
    exit()

# --- CONFIGURATION ---
TRAIL_LENGTH = 0

def process_video_strongsort_pp(args):
    source = args['source']
    count_ = args['count']
    batch_mode = args['batch']
    target_classes = args['classes']
    model_path = args['model']    
    conf_threshold = args['conf']
    nosave = args['nosave']

    print(f"Loading Model: {model_path} with Conf: {conf_threshold}")
    try:
        model = YOLO(model_path)
        names = model.names
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return
    
    # Initialize StrongSORT
    tracker = StrongSort(
        reid_weights=Path('mobilenetv2_x1_4.pt'), 
        device='cuda:0', 
        half=False,
        max_age=30,
        n_init=5,
        min_conf=0.5,
        max_cos_dist=0.5,
        max_iou_dist=0.7,
    )

    # --- PER-CLASS ID MAPPING VARIABLES ---
    trajectories = {}
    class_id_mappings = {}  # Stores {class_id: {raw_strongsort_id: clean_id}}
    class_counters = {}     # Stores {class_id: next_available_id}
    # --------------------------------------

    output_dir = 'output'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    base_input_name = os.path.splitext(os.path.basename(source))[0] if source != '0' else 'webcam'
    
    # --- FILE NAMING LOGIC ---
    base_name_no_ext = os.path.join(output_dir, f'{base_input_name}_strongsort_pp')
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

    # Only initialize VideoWriter if save_vid is True
    out = None
    if not nosave:
        try: fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except: fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        
        # Zero Overlap Logic (Strict NMS)
        results = model.predict(frame, conf=conf_threshold, iou=0.5, classes=target_classes, verbose=False)
        
        t2 = time.time()
        
        inference_ms = (t2 - t1) * 1000
        if inference_ms > 0: model_fps_str = f"Model FPS: {(1000.0 / inference_ms):.1f}"

        dets_to_track = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)
                dets_to_track.append([x1, y1, x2, y2, conf, cls])
        
        dets_to_track = np.array(dets_to_track)
        if len(dets_to_track) == 0: dets_to_track = np.empty((0, 6))

        # Update Tracker
        tracked_objects = tracker.update(dets_to_track, frame)
        object_count = len(tracked_objects)
        
        # Cleanup Trajectories
        current_raw_ids = [int(obj[4]) for obj in tracked_objects]
        for key in list(trajectories.keys()):
            if key not in current_raw_ids:
                del trajectories[key]

        # --- WRITE LABELS & DRAW ---
        with open(final_text_path, 'a') as txt_file:
            for output in tracked_objects:
                x1, y1, x2, y2, raw_id, conf, cls, index = output
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                raw_id = int(raw_id)
                cls_id = int(cls)
                
                # --- NEW PER-CLASS ID MAPPING ---
                if cls_id not in class_counters:
                    class_counters[cls_id] = 1
                    class_id_mappings[cls_id] = {}
                
                if raw_id not in class_id_mappings[cls_id]:
                    class_id_mappings[cls_id][raw_id] = class_counters[cls_id]
                    class_counters[cls_id] += 1
                
                visual_id = class_id_mappings[cls_id][raw_id]
                # -------------------------------

                # Write to CSV
                class_name = names.get(cls_id, "Unknown")
                # Format: Frame,Timestamp,Class,ID,Conf,x1,y1,x2,y2
                line = f"{frameId},{timestamp_str},{class_name},{int(visual_id)},{float(conf):.2f},{x1},{y1},{x2},{y2}\n"
                txt_file.write(line)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                if raw_id not in trajectories:
                    trajectories[raw_id] = deque(maxlen=TRAIL_LENGTH)
                trajectories[raw_id].append((center_x, center_y))

                # Color generation (Mixing class ID into seed so Person 1 != Car 1)
                color_seed = visual_id + (cls_id * 50)
                color = [int(c) for c in id_colors[color_seed % len(id_colors)]]
                
                label = f"{class_name}: {visual_id} {conf:.2f}"

                for i in range(1, len(trajectories[raw_id])):
                    if trajectories[raw_id][i - 1] is None or trajectories[raw_id][i] is None: continue
                    thickness = int(np.sqrt(64 / float(len(trajectories[raw_id]) - i + 1)) * 2)
                    cv2.line(frame, trajectories[raw_id][i - 1], trajectories[raw_id][i], color, thickness)

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

        # --- HUD (Display Stats) ---
        if not nosave or not batch_mode:
            cv2.putText(frame, real_fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, model_fps_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # --- PER CLASS COUNT DISPLAY ---
            if count_:
                y_offset = 30
                num_classes = len(class_counters)
                # Height of black box depends on how many classes we found
                box_h = 30 * (num_classes + 1) if num_classes > 0 else 40
                
                # Draw black background
                cv2.rectangle(frame, (w - 220, 0), (w, box_h), (0,0,0), -1)
                cv2.putText(frame, "Total Counts:", (w - 210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if num_classes == 0:
                     cv2.putText(frame, "0", (w - 210, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    for cls_id, count_val in class_counters.items():
                        y_offset += 30
                        # count_val is 'next_id', so total found is count_val - 1
                        actual_count = count_val - 1
                        cls_name = names.get(cls_id, str(cls_id))
                        if len(cls_name) > 10: cls_name = cls_name[:10]
                        
                        text = f"{cls_name}: {actual_count}"
                        cv2.putText(frame, text, (w - 210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # -------------------------------

        if not nosave and out is not None:
            out.write(frame)

        if not batch_mode:
            cv2.imshow(f"StrongSort_{source}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if out is not None: out.release()
    if not batch_mode: cv2.destroyAllWindows()
    print(f"[{base_input_name}] Finished.")
    print(f"[{base_input_name}] Labels saved to {final_text_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', nargs='+', type=str, default=['0'], help='Video file or Folder')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--track', action='store_true', default=True)
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int, default=[0])
    
    # --- NEW ARGUMENT ---
    parser.add_argument('--nosave', action='store_true', help='Do not save output video file')
    
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
        'count': args.count, 
        'batch': args.batch, 
        'classes': args.classes,
        'model': args.model, 
        'conf': args.conf, 
        'nosave': args.nosave
    } for s in video_sources]
    
    # --- SEQUENTIAL PROCESSING ---
    for i, args_dict in enumerate(process_args_list):
        print(f"\n--- Video {i+1} of {len(process_args_list)} ---")
        process_video_strongsort_pp(args_dict)
    
    print("\nAll processing complete.")