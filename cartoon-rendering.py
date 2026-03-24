import cv2
import numpy as np

img = cv2.imread('ferret.jpeg')

# 경계는 유지하면서 내부 색을 단색에 가깝게 뭉갬
color = img.copy()
for _ in range(10):
    color = cv2.bilateralFilter(color, d=9, sigmaColor=200, sigmaSpace=200)

# 만화 특유의 선명하고 진한 색감
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.2, 0, 255)   # 채도 2.2배
hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)   # 명도 살짝 올림
color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# 굵고 선명한 윤곽선 추출 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 7)

# Adaptive Threshold로 세밀한 윤곽선
edges_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 2)

# Canny로 굵은 외곽선 추가
edges_canny = cv2.Canny(gray, 30, 100)
kernel = np.ones((2, 2), np.uint8)
edges_canny = cv2.dilate(edges_canny, kernel, iterations=1)  # 선 굵게
edges_canny_inv = cv2.bitwise_not(edges_canny)

# 두 에지 합치기 (AND → 둘 다 선인 곳만 선으로)
edges = cv2.bitwise_and(edges_adapt, edges_canny_inv)

# 색 + 윤곽선 합성 
cartoon = cv2.bitwise_and(color, color, mask=edges)

# 경계를 한 번 더 또렷하게
blurred = cv2.GaussianBlur(cartoon, (0, 0), 3)
cartoon = cv2.addWeighted(cartoon, 1.5, blurred, -0.5, 0)

# ── 원본과 나란히 비교 ────────────────────────────────────────────────
h, w = img.shape[:2]

# 화면에 맞게 축소
max_w = 700
if w > max_w:
    scale = max_w / w
    img     = cv2.resize(img,     (int(w * scale), int(h * scale)))
    cartoon = cv2.resize(cartoon, (int(w * scale), int(h * scale)))
    h, w = img.shape[:2]

# 구분선
divider = np.full((h, 4, 3), 255, dtype=np.uint8)
combined = np.hstack([img, divider, cartoon])

# 라벨
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(combined, "Original", (10, 30),
            font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(combined, "Cartoon",  (w + 14, 30),
            font, 0.9, (100, 220, 100), 2, cv2.LINE_AA)

# Display
cv2.imshow("Original  vs  Cartoon", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save
cv2.imwrite('result.jpg', cartoon)
print("저장 완료: result.jpg")
