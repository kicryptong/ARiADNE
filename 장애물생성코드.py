from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 이미지 불러오기
image_path = "1.png"  # 원본 파일 이름
output_path = "baekyang_obs1.png"  # 저장할 파일 이름
image = Image.open(image_path).convert("RGB")

# 박스 좌표 및 크기 (직사각형)
yellow_boxes = [(224, 195)]
box_width = 12
box_height = 32

# 노란 박스 그리기
draw = ImageDraw.Draw(image)
for x, y in yellow_boxes:
    draw.rectangle([x, y, x + box_width - 1, y + box_height - 1], fill=(127, 127, 127))

# 이미지 저장
image.save(output_path) # 저장할 때 활성화

# 이미지 시각화 (matplotlib)
plt.imshow(image)
plt.axis("off")
plt.title("저장된 이미지 (노란 박스 포함)")
plt.show()

##### 박스 좌표 및 크기 (직사각형) 수정하면 장애물생성임.