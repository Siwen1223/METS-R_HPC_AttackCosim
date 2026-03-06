import os
import cv2
import matplotlib.pyplot as plt

vid = 1                     # vehicle id to replay
save_folder = "out_0305"
folder = save_folder+f"/{vid}/camera"
start_frame = 1335
end_frame = 1800
pause_time = 0.5 

frames = [
    os.path.join(folder, f"im{frame:08d}.png")
    for frame in range(start_frame, end_frame + 1)
    if os.path.exists(os.path.join(folder, f"im{frame:08d}.png"))
]

if not frames:
    print("No camera images found for vehicle", vid)
    exit()

plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))

try:
    for im_path in frames:
        img = cv2.imread(im_path)
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")
        frame_number = os.path.basename(im_path).split("im")[1].split(".png")[0]
        ax.set_title(f"RGB Camera data from vehicle {vid} - Frame {int(frame_number)}")

        plt.pause(pause_time)
        ax.cla()

except KeyboardInterrupt:
    print("\nPlayback stopped by user.")

plt.ioff()
plt.show()
