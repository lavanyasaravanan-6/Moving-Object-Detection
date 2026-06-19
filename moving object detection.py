import cv2

# Start webcam
cap = cv2.VideoCapture(0)

# Read first frame
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():

    # Find difference between frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Blur image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate image
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        # Ignore small movements
        if cv2.contourArea(contour) < 1000:
            continue

        # Draw rectangle around moving object
        (x, y, w, h) = cv2.boundingRect(contour)

        cv2.rectangle(
            frame1,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame1,
            "Moving Object",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show video
    cv2.imshow("Moving Object Detection", frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # Exit when pressing Q
    if cv2.waitKey(40) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()