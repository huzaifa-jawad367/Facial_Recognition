from deepface import DeepFace
import matplotlib.pyplot as plt
import time

def detect_faces(img_path, backend="mtcnn"):
    print(f"\n––– Backend: {backend} –––")
    tic = time.time()
    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=backend,
            align=True,
            enforce_detection=False   # ← don’t error if no face
        )
    except Exception as e:
        print("  Exception:", e)
    toc = time.time()

    if not faces:
        print("  → no faces returned")

    return faces, toc-tic
    
    
if __name__ == '__main__':
    backends = ['opencv','ssd','dlib','mtcnn','fastmtcnn','retinaface','mediapipe','yolov8','yunet','centerface']
    img_path = 'Face Recognition Dataset-1/Dr. Adnan ul Hasan/8b5afb0e-9244-483a-95a4-713e6ff47662.jpg'

    for backend in backends:
        
        faces, runtime = detect_faces(img_path, backend=backend)

        # if faces is a list of dicts:
        # face_arr = faces[0]['face'][..., ::-1]  # convert BGR→RGB
        face_arr = faces[1]['face']  # convert BGR→RGB
        plt.imshow(face_arr)
        plt.axis('off')
        plt.title(f"{backend}: {len(faces)} faces in {runtime:.2f}s")
        plt.show()