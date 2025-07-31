import random
import sys
import string

def generateWatermark(min_len=5, max_len=256):
    # ASCII letters + space
    chars = [
        string.ascii_letters + ' ' + string.digits + string.punctuation,
        string.ascii_letters + ' ' + string.digits,
        string.ascii_letters + ' ',
        string.ascii_letters,
        '01234567890',
        ' !@#$%^&*()_+-=[]{}|;:\'",.<>?/',
        'abcdefghijklmnopqrstuvwxyz',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    ]
    sentence = ''.join(random.choices(random.choice(chars), k=random.randint(min_len, max_len)))
    return sentence

def saveWatermarks(filename, count=10000):
    with open(filename, 'a', encoding='latin1') as f:
        for _ in range(count):
            watermark = generateWatermark()
            if isinstance(watermark, bytes):
                # convert bytes to latin1 string for saving
                watermark = watermark.decode('latin1')
            f.write(watermark + '\n')
    print(f"Saved {count} watermarks to {filename}")

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "watermarks.txt"
    saveWatermarks(filename, count=1082)
