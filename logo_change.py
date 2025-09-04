from PIL import Image

# 변환할 PNG 파일 경로
png_file = "icon-512.png"

# 저장할 ICO 파일 경로
ico_file = "icon.ico"

# PNG 열기
img = Image.open(png_file)

# ICO로 저장 (Windows용, 여러 해상도 포함 가능)
img.save(ico_file, format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])

print(f"{png_file} → {ico_file} 변환 완료!")
