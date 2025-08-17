# GitFlow Workflow

## Giới thiệu
GitFlow là một mô hình quản lý nhánh Git phổ biến, giúp tổ chức quy trình phát triển phần mềm một cách rõ ràng và hiệu quả. Mô hình này đặc biệt hữu ích cho các dự án có nhiều thành viên, cần quản lý các tính năng, bản phát hành và sửa lỗi một cách có hệ thống.

## Các nhánh chính
- **`main`**: Chứa mã nguồn ổn định, sẵn sàng triển khai lên môi trường production.
- **`develop`**: Tích hợp các tính năng mới, được xem là phiên bản ổn định để chuẩn bị cho bản phát hành tiếp theo.

## Các nhánh phụ
- **`feature/`**: Dùng để phát triển các tính năng mới.  
  - **Tạo từ**: Nhánh `develop`.  
  - **Merge vào**: Nhánh `develop`.  
  - **Quy tắc đặt tên**: `feature/<tên-tính-năng>` (VD: `feature/login-google`).  
- **`release/`**: Chuẩn bị cho bản phát hành, bao gồm sửa lỗi nhỏ và cập nhật phiên bản.  
  - **Tạo từ**: Nhánh `develop`.  
  - **Merge vào**: Cả `main` và `develop`.  
  - **Quy tắc đặt tên**: `release/<phiên-bản>` (VD: `release/1.0.0`).  
- **`hotfix/`**: Sửa lỗi khẩn cấp trên môi trường production.  
  - **Tạo từ**: Nhánh `main`.  
  - **Merge vào**: Cả `main` và `develop`.  
  - **Quy tắc đặt tên**: `hotfix/<mô-tả-lỗi>` (VD: `hotfix/fix-login-bug`).

## Quy trình làm việc
1. **Phát triển tính năng mới**:  
   - Tạo nhánh `feature` từ `develop`.  
   - Sau khi hoàn thành, merge nhánh `feature` vào `develop` và xóa nhánh `feature`.  
2. **Chuẩn bị phát hành**:  
   - Tạo nhánh `release` từ `develop`.  
   - Thực hiện các chỉnh sửa nhỏ (nếu cần) và cập nhật phiên bản.  
   - Merge nhánh `release` vào `main` và `develop`, sau đó gắn tag phiên bản và xóa nhánh `release`.  
3. **Sửa lỗi khẩn cấp**:  
   - Tạo nhánh `hotfix` từ `main`.  
   - Sửa lỗi, sau đó merge nhánh `hotfix` vào cả `main` và `develop`, rồi xóa nhánh `hotfix`.

## Ví dụ lệnh GitFlow
### 1. Phát triển tính năng mới
```bash
git checkout develop
git checkout -b feature/login-google
# Viết mã, commit thay đổi
git push origin feature/login-google
# Tạo Pull Request trên GitHub từ feature/login-google vào develop
# Sau khi PR được phê duyệt và kiểm tra bởi GitHub Actions, merge PR trên GitHub
git checkout develop
git pull origin develop
git branch -d feature/login-google
```

### 2. Chuẩn bị phát hành
```bash
git checkout develop
git checkout -b release/1.0.0
# Thực hiện các chỉnh sửa nhỏ, cập nhật phiên bản
git push origin release/1.0.0
# Tạo Pull Request trên GitHub từ release/1.0.0 vào main
# Sau khi PR được phê duyệt và kiểm tra bởi GitHub Actions, merge PR trên GitHub
git checkout main
git pull origin main
git tag -a v1.0.0 -m "Release 1.0.0"
# Tạo Pull Request trên GitHub từ release/1.0.0 vào develop
# Sau khi PR được phê duyệt và kiểm tra bởi GitHub Actions, merge PR trên GitHub
git checkout develop
git pull origin develop
git branch -d release/1.0.0
```

### 3. Sửa lỗi khẩn cấp
```bash
git checkout main
git checkout -b hotfix/fix-login-bug
# Sửa lỗi, commit thay đổi
git push origin hotfix/fix-login-bug
# Tạo Pull Request trên GitHub từ hotfix/fix-login-bug vào main
# Sau khi PR được phê duyệt và kiểm tra bởi GitHub Actions, merge PR trên GitHub
git checkout main
git pull origin main
# Tạo Pull Request trên GitHub từ hotfix/fix-login-bug vào develop
# Sau khi PR được phê duyệt và kiểm tra bởi GitHub Actions, merge PR trên GitHub
git checkout develop
git pull origin develop
git branch -d hotfix/fix-login-bug
```

## Lưu ý
- **Công cụ hỗ trợ**: Có thể sử dụng các công cụ như `git-flow` (một tiện ích dòng lệnh) để tự động hóa quy trình GitFlow.
- **Xung đột merge**: Luôn kiểm tra và giải quyết xung đột khi merge nhánh để đảm bảo mã nguồn ổn định.
- **Sử dụng Pull Request với GitHub Actions**:  
  - Thay vì merge trực tiếp các nhánh `feature`, `release`, hoặc `hotfix`, hãy tạo **Pull Request (PR)** trên GitHub để xem xét mã nguồn trước khi merge.  
  - K kết hợp **GitHub Actions** để tự động hóa quy trình kiểm tra:  
    - Tạo file workflow (VD: `.github/workflows/ci.yml`) để chạy kiểm tra mã nguồn (unit tests, linters) mỗi khi có PR.  
    - Ví dụ: Một GitHub Action có thể chạy `npm test` hoặc `flake8` để kiểm tra mã JavaScript hoặc Python.  
    - Cấu hình PR để yêu cầu ít nhất một người phê duyệt trước khi merge vào `develop` hoặc `main`.  
  - Lợi ích: Đảm bảo chất lượng mã, giảm lỗi, và tăng tính minh bạch trong quy trình phát triển.  
- **Tài liệu tham khảo**:  
  - [GitFlow by Vincent Driessen](https://nvie.com/posts/a-successful-git-branching-model/)