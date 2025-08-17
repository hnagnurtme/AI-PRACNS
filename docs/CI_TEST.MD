# CI Test Workflow

## Mục đích
File workflow `CI Test` được thiết kế để tự động hóa việc chạy các bài kiểm tra (tests) mỗi khi có **Pull Request (PR)** được tạo hoặc cập nhật nhắm đến các nhánh `develop` hoặc `main`. Workflow này là một phần của quy trình GitFlow, đảm bảo chất lượng mã nguồn trước khi merge, giúp giảm lỗi và tăng tính minh bạch trong quá trình phát triển.

## Cách hoạt động
Workflow này được kích hoạt khi:
- Một Pull Request được tạo hoặc cập nhật nhắm đến nhánh `develop` hoặc `main`.
- Workflow chạy trên môi trường `ubuntu-latest`, thực hiện các bước sau:
  1. Kiểm tra mã nguồn từ repository.
  2. Cấp quyền thực thi cho script kiểm tra (`scripts/test.sh`).
  3. Chạy script kiểm tra để thực hiện các bài kiểm tra.

## Cấu trúc file workflow
File workflow được định nghĩa trong `.github/workflows/ci.yml` với nội dung như sau:
```yaml
name: CI Test

on:
  pull_request:
    branches: 
      - develop
      - main

jobs:
  run-tests:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout source code
        uses: actions/checkout@v3

      - name: Make test script executable
        run: chmod +x scripts/test.sh

      - name: Run test script
        run: ./scripts/test.sh
```

## Các bước chi tiết
1. **Checkout source code**: Sử dụng action `actions/checkout@v3` để tải mã nguồn từ repository, bao gồm các thay đổi trong Pull Request.
2. **Make test script executable**: Cấp quyền thực thi cho file `scripts/test.sh` để đảm bảo script có thể chạy được trên môi trường Ubuntu.
3. **Run test script**: Thực thi script `scripts/test.sh` để chạy các bài kiểm tra (unit tests, linters, hoặc các kiểm tra khác tùy thuộc vào nội dung script).

## Yêu cầu
- Repository phải có file `scripts/test.sh` với các lệnh kiểm tra phù hợp (ví dụ: chạy `npm test` cho JavaScript, `pytest` cho Python, hoặc các công cụ linter như `flake8`).
- Script `test.sh` cần được cấu hình để trả về mã thoát (exit code) `0` nếu kiểm tra thành công, hoặc mã khác nếu thất bại, để GitHub Actions có thể xác định trạng thái của job.

## Hướng dẫn sử dụng
1. Đảm bảo file `scripts/test.sh` tồn tại trong repository và chứa các lệnh kiểm tra phù hợp với dự án của bạn.
2. Đẩy file workflow `.github/workflows/ci.yml` lên repository.
3. Tạo Pull Request nhắm đến nhánh `develop` hoặc `main` để kích hoạt workflow.
4. Kiểm tra kết quả chạy workflow trong tab **Actions** trên GitHub.
5. Nếu workflow thất bại, xem log để xác định lỗi và sửa đổi trước khi merge PR.

## Lưu ý
- **Tùy chỉnh script kiểm tra**: Điều chỉnh nội dung `scripts/test.sh` dựa trên ngôn ngữ lập trình hoặc công cụ kiểm tra của dự án (ví dụ: `npm test` cho Node.js, `pytest` cho Python, hoặc `rubocop` cho Ruby).
- **Bảo mật**: Đảm bảo script `test.sh` không chứa các lệnh nhạy cảm hoặc truy cập tài nguyên không an toàn.
- **Tích hợp với GitFlow**: Workflow này hỗ trợ quy trình GitFlow bằng cách kiểm tra mã nguồn trong các nhánh `feature`, `release`, hoặc `hotfix` trước khi merge vào `develop` hoặc `main`.
- **Tài liệu tham khảo**:
  - [GitHub Actions Documentation](https://docs.github.com/en/actions)
  - [GitFlow by Vincent Driessen](https://nvie.com/posts/a-successful-git-branching-model/)