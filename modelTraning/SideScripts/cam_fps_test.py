import cv2

# Ścieżka do pliku wideo
video_path = 'twoje_video.mp4'

# Otwarcie pliku wideo
cap = cv2.VideoCapture(0)

# Sprawdzenie, czy wideo zostało poprawnie załadowane
if not cap.isOpened():
    print("Nie można otworzyć pliku wideo.")
else:
    # Pobranie wartości FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Liczba klatek na sekundę (FPS): {fps}")

# Zwolnienie zasobu
cap.release()