import re
import os

CONFIG_PATH = "eztalk_proxy/core/config.py"

def increment_version(version_str):
    # 匹配版本号格式: 1.9.9.77-gcs-support 或 1.9.9.77
    # 我们只增加最后一个数字
    
    # 查找主要版本号部分
    match = re.match(r"(\d+\.\d+\.\d+\.)(\d+)(.*)", version_str)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        suffix = match.group(3)
        
        # 增加版本号
        new_number = number + 1
        new_version = f"{prefix}{new_number}{suffix}"
        return new_version
    
    # 如果格式不匹配，尝试简单的 x.x.x 格式
    match_simple = re.match(r"(\d+\.\d+\.)(\d+)(.*)", version_str)
    if match_simple:
        prefix = match_simple.group(1)
        number = int(match_simple.group(2))
        suffix = match_simple.group(3)
        
        new_number = number + 1
        new_version = f"{prefix}{new_number}{suffix}"
        return new_version
        
    print(f"Warning: Could not parse version string '{version_str}'. No changes made.")
    return version_str

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 查找 APP_VERSION 行
    # APP_VERSION = os.getenv("APP_VERSION", "1.9.9.77-gcs-support")
    pattern = r'(APP_VERSION = os\.getenv\("APP_VERSION", ")(.*?)("\))'
    
    match = re.search(pattern, content)
    if match:
        current_version = match.group(2)
        new_version = increment_version(current_version)
        
        if new_version != current_version:
            new_content = content.replace(match.group(0), f'{match.group(1)}{new_version}{match.group(3)}')
            
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(new_content)
                
            print(f"Version updated: {current_version} -> {new_version}")
        else:
            print("Version not updated (format mismatch or same version).")
    else:
        print("Error: APP_VERSION definition not found in config file.")

if __name__ == "__main__":
    main()