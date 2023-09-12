import cv2
import numpy as np
import matplotlib.pyplot as plt

# ảnh đầu vào
img_path = "gear7.jpg"
img = cv2.imread(img_path)
# chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Lọc ra các pixel còn lại
gray = cv2.medianBlur(gray, 5)

# Nhị phân hóa hình ảnh thang độ xám bằng phương pháp của Otsu để tạo ngưỡng hình ảnh tự động
(thres, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

floodfill = bw.copy()

# FloodFill: lấp đầy các vùng trống của bánh răng
h, w = bw.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Flood fill bắt đầu từ điểm có tọa độ (0, 0)
cv2.floodFill(floodfill, mask, (0, 0), 0);

# đảo màu
floodfill_inv = cv2.bitwise_not(floodfill)

# cộng 2 ảnh bw và floodfill_inv
fill = bw & floodfill_inv

# đảo màu
fill = cv2.bitwise_not(fill)
# Kernel size is (100. 100), determined empirically
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(100, 100))
# Open region
open = cv2.morphologyEx(fill, cv2.MORPH_OPEN, kernel)

overlap = cv2.addWeighted(fill, 0.5, open, 0.5, 0.0)

# Tính đường kính ngoài và trong
outer_diameter = 0
inner_diameter = 0
contours1, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours1:
    (x, y, w, h) = cv2.boundingRect(contour)
    diameter1 = w if w > h else h
    if diameter1 > outer_diameter:
        outer_diameter = diameter1
contours2, _ = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours2:
    (x, y, w, h) = cv2.boundingRect(contour)
    diameter2 = w if w > h else h
    if diameter2 > inner_diameter:
        inner_diameter = diameter2

# phân đoạn thanh răng
img_cogs = open | bw

# đảo ngược màu
img_cogs = cv2.bitwise_not(img_cogs)
# tìm các khu vực là răng
count, labels, stats, centroids = cv2.connectedComponentsWithStats(img_cogs)
# danh sách các khu vực được tìm thấy
areas = stats[:, 4]
# Lấy trọng tâm của area (để phát hiện răng khuyết)
median = np.median(areas)

num_teeth = 0
contours, _ = cv2.findContours(img_cogs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    # Kiểm tra kích thước hình chữ nhật bao quanh contour
    if w >= 5 and h >= 5:
        num_teeth += 1
print(num_teeth)
# Đếm số răng
img_centroids = img.copy()
cog_count = 0
for i in range(0, count):
    if areas[i] > median - 100 and areas[i] < median + 100:
        cv2.circle(img_centroids, (int(centroids[i, 0]), int(centroids[i, 1])), 0, (0, 255, 0), 5)
        cog_count += 1

# modul:
m = outer_diameter/(num_teeth+2)
arr = [0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 60, 80, 100]
i = 0
while i < 25:
    if arr[i] <= m <= arr[i+1] :
        if m-arr[i] < arr[i+1]-m :
            m = arr[i]
        else:
            m = arr[i+1]
    i = i+1
# bước răng :
p = round(3.14*m, 3)

fig = plt.figure(figsize=(16, 9))
ax1, ax2 = fig.subplots(1, 2)
ax1.imshow (img, cmap = 'gray')
ax2.imshow (img_centroids, cmap = 'gray')
ax1.set_title ("Ảnh gốc", fontsize=15)
ax2.set_title (f"Số răng : {cog_count}, số răng hỏng: {num_teeth - cog_count}\nModul: {m}, bước răng: {p} \nĐường kính ngoài: {outer_diameter} \nĐường kính trong: {inner_diameter}", fontsize=15)
plt.show()