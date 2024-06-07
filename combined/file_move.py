import os
import shutil

# 원본 디렉토리와 대상 디렉토리 경로를 설정합니다.
source_dir = "/store/cpnr/users/yewzzang/KNO_mu_500/h5_wall2"  # 원본 폴더 경로로 변경
destination_dir = "/store/cpnr/users/yewzzang/KNO_mu_500/h5_wall2_sub"  # 대상 폴더 경로로 변경

# 파일 이름 패턴을 생성합니다.
for i in range(1001, 2001):
    file_name = f"mu_500MeV_{i}.h5"
    source_file = os.path.join(source_dir, file_name)
    destination_file = os.path.join(destination_dir, file_name)
    
    # 파일이 존재하는지 확인하고 이동합니다.
    if os.path.exists(source_file):
        shutil.move(source_file, destination_file)
        print(f"Moved: {source_file} -> {destination_file}")
    else:
        print(f"File not found: {source_file}")