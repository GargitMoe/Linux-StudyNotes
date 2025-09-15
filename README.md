# Linux 基础命令入门

---

## 1. `ls` —— 查看目录内容

- 用途：列出当前目录下的文件和文件夹。
- 常见用法：
  ```bash
  ls        # 列出文件名
  ls -l     # 列出详细信息（权限、大小、时间）
  ls -a     # 包括隐藏文件
  ls -lh    # 以人类可读的方式显示大小

## 2. `cd` —— 切换目录

- 用途：进入或者切换目录
- 常见用法：
  ```bash
  cd Documents           # 进入当前目录下的 Documents 子目录
  cd /home/user/Documents # 使用绝对路径进入目录
  cd ..                  # 返回上一级目录
  cd ../..               # 返回上上级目录

## 3. `pwd` —— 查看目前目录

- 用途：查看目前目录，实际上全称叫：print working directory
- 常见用法：
  ```bash
  cd Documents           # 进入当前目录下的 Documents 子目录
  cd /home/user/Documents # 使用绝对路径进入目录
  cd ..                  # 返回上一级目录
  cd ../..               # 返回上上级目录
  打印后可能出现：/home/user/Documents
