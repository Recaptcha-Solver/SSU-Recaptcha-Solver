import asyncio
import setting
from solverecaptchas.solver import Solver


if __name__ == '__main__':
     print("-------모드 선택-------")
     print("1 : OpenCV 테스트")
     print("나머지 : 실행")
     arg = input()
     proxy = "none"
     if proxy.lower() == "none":
          proxy = None
     client = Solver(setting.pageurl, setting.sitekey, proxy=proxy)
     if (arg == "1"):
          img_path = input("이미지 경로 : ")
          image_class = input("이미지 종류 (ex : \"bus\") : ")
          result = asyncio.run(client.start_test(img_path=img_path,image_class=image_class))
          if result:
               print(result)
          exit(0)
     else: # 실행
          result = asyncio.run(client.start())
          if result:
               print(result)

