import os
import webbrowser

import numpy as np
from multiprocessing import Process, Array, cpu_count, current_process, Lock
from flask import Flask, request, render_template
import time
import json
from gevent import pywsgi
from multiprocessing import freeze_support

app = Flask(__name__)


def fivProIncC(tG):
    if tG <= 76:
        return 0
    return (tG - 75) * 60


def fivProIncCf(tG):
    if tG == 8:
        return 516
    return 0


def fivProIncL(tG):
    if tG <= 66:
        return 0
    return (tG - 65) * 70


def fivProIncLf(tG):
    if tG == 7:
        return 466
    if tG == 8:
        return 866
    return 0


def gachaOnceC(starburst, cFiveG, cFourG, cL55):
    if cFiveG == 89:
        cFiveG = 0
        cFourG = 0
        starburst += 40
        if cL55 == 0:  # 小保底
            theNum = np.random.randint(1, 17)
            if theNum <= 9:
                cL55 = 0
                return 51, starburst, cFiveG, cFourG, cL55  # 没歪
            else:
                cL55 = 1
                return 52, starburst, cFiveG, cFourG, cL55  # 歪了
        cL55 = 0
        return 51, starburst, cFiveG, cFourG, cL55
    theNum = np.random.randint(1, 1001)
    if cFourG == 9:
        if theNum <= 6 + fivProIncC(cFiveG):
            cFiveG = 0
            cFourG = 0
            starburst += 40
            if cL55 == 0:  # 小保底
                theNum = np.random.randint(1, 17)
                if theNum <= 9:
                    cL55 = 0
                    return 51, starburst, cFiveG, cFourG, cL55  # 没歪
                else:
                    cL55 = 1
                    return 52, starburst, cFiveG, cFourG, cL55  # 歪了
            cL55 = 0
            return 51, starburst, cFiveG, cFourG, cL55
        else:
            cFiveG += 1
            cFourG = 0
            starburst += 10  # 无法确定四星是否满命，保守取平均值10个
            return 4, starburst, cFiveG, cFourG, cL55
    if theNum <= 6 + fivProIncC(cFiveG):
        cFiveG = 0
        starburst += 40
        if cL55 == 0:  # 小保底
            theNum = np.random.randint(1, 17)
            if theNum <= 9:
                cL55 = 0
                return 51, starburst, cFiveG, cFourG, cL55  # 没歪
            else:
                cL55 = 1
                return 52, starburst, cFiveG, cFourG, cL55  # 歪了
        cL55 = 0
        return 51, starburst, cFiveG, cFourG, cL55
    if theNum >= 944 - fivProIncCf(cFourG):
        cFiveG += 1
        cFourG = 0
        starburst += 8
        return 4, starburst, cFiveG, cFourG, cL55
    else:
        cFiveG += 1
        cFourG += 1
        return 3, starburst, cFiveG, cFourG, cL55


def gachaOnceL(starburst, lFiveG, lFourG, lL55):
    if lFiveG == 79:
        lFiveG = 0
        lFourG = 0
        starburst += 40
        if lL55 == 0:  # 小保底
            theNum = np.random.randint(1, 5)
            if theNum < 4:
                lL55 = 0
                return 51, starburst, lFiveG, lFourG, lL55  # 没歪
            else:
                lL55 = 1
                return 52, starburst, lFiveG, lFourG, lL55  # 歪了
        lL55 = 0
        return 51, starburst, lFiveG, lFourG, lL55
    theNum = np.random.randint(1, 1001)
    if lFourG == 9:
        if theNum <= 6 + fivProIncL(lFiveG):
            lFiveG = 0
            lFourG = 0
            starburst += 40
            if lL55 == 0:  # 小保底
                theNum = np.random.randint(1, 5)
                if theNum < 4:
                    lL55 = 0
                    return 51, starburst, lFiveG, lFourG, lL55  # 没歪
                else:
                    lL55 = 1
                    return 52, starburst, lFiveG, lFourG, lL55  # 歪了
            lL55 = 0
            return 51, starburst, lFiveG, lFourG, lL55
        else:
            lFiveG += 1
            lFourG = 0
            starburst += 10  # 无法确定四星是否满命，保守取平均值10个
            return 4, starburst, lFiveG, lFourG, lL55
    if theNum <= 8 + fivProIncC(lFiveG):
        lFiveG = 0
        starburst += 40
        if lL55 == 0:  # 小保底
            theNum = np.random.randint(1, 5)
            if theNum < 4:
                lL55 = 0
                return 51, starburst, lFiveG, lFourG, lL55  # 没歪
            else:
                lL55 = 1
                return 52, starburst, lFiveG, lFourG, lL55  # 歪了
        lL55 = 0
        return 51, starburst, lFiveG, lFourG, lL55
    if theNum >= 934 - fivProIncLf(lFourG):
        lFiveG += 1
        lFourG = 0
        starburst += 8
        return 4, starburst, lFiveG, lFourG, lL55
    else:
        lFiveG += 1
        lFourG += 1
        return 3, starburst, lFiveG, lFourG, lL55


def letsGo(tics, caras, lighs, count, cL55, lL55):
    starburst = 0
    cFiveG = 0
    cFourG = 0
    lFiveG = 0
    lFourG = 0
    cara0 = caras.copy()
    ligh0 = lighs.copy()
    lighR = []
    caraR = []
    for i in range(count):
        lighR.append(0)
        caraR.append(0)
    for i in range(count):
        tics += starburst // 20
        starburst = starburst % 20
        while cara0[i] > 0 and tics > 0:
            res0 = gachaOnceC(starburst, cFiveG, cFourG, cL55)
            starburst = res0[1]
            cFiveG = res0[2]
            cFourG = res0[3]
            cL55 = res0[4]
            tics -= 1
            if res0[0] == 51:
                caraR[i] += 1
                cara0[i] -= 1
            tics += starburst // 20
            starburst = starburst % 20
        while ligh0[i] > 0 and tics > 0:
            res1 = gachaOnceL(starburst, lFiveG, lFourG, lL55)
            starburst = res1[1]
            lFiveG = res1[2]
            lFourG = res1[3]
            lL55 = res1[4]
            tics -= 1
            if res1[0] == 51:
                lighR[i] += 1
                ligh0[i] -= 1
            tics += starburst // 20
            starburst = starburst % 20
    return caraR, lighR


def worker(start, end, tic, cara, ligh, caraRT, lighRT, count, lock, cL55, lL55):
    print(f"\r进程 {os.getpid()} 启动...", end="")
    lighT = []
    caraT = []
    for i in range(count):
        lighT.append(0)
        caraT.append(0)
    for i in range(start, end):
        res = letsGo(tic, cara, ligh, count, cL55, lL55)
        for j in range(count):
            if res[0][j] == cara[j] and cara[j] != 0:
                caraT[j] += 1
            if res[1][j] == ligh[j] and ligh[j] != 0:
                lighT[j] += 1
    with lock:  # 使用锁来保护共享数据的更新
        for j in range(count):
            caraRT[j] += caraT[j]
            lighRT[j] += lighT[j]


def main(tic, count, cara, ligh, cL55, lL55):
    total_range = 200000  # 二十万
    print('总共{}发抽卡，按照：'.format(tic))
    for i in range(count):
        if cara[i] != 0:
            print('角色{}只，'.format(cara[i]), end='')
        if ligh[i] != 0:
            print('光锥{}张，'.format(ligh[i]), end='')
    print('的顺序抽卡')
    num_processes = cpu_count() * 2 - 4  # 或者任何你想要的进程数
    print('总模拟次数：{}'.format(total_range))
    print('四星返利比例为：8星辉/四星（即不考虑四星满命反利）')
    print()
    print('使用{}个CPU核心模拟寄算中……'.format(cpu_count()))
    print()
    # 使用 multiprocessing.Array 作为共享内存来存储结果
    caraRT = Array('i', np.zeros((count,), dtype=int))
    lighRT = Array('i', np.zeros((count,), dtype=int))
    lock = Lock()  # 创建一个锁
    time0 = time.time()

    processes = []
    range_per_process = total_range // num_processes

    for index in range(num_processes):
        start = index * range_per_process
        end = start + range_per_process if index < num_processes - 1 else total_range
        p = Process(target=worker, args=(start, end, tic, cara, ligh, caraRT, lighRT, count, lock, cL55, lL55))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    print()
    print()
    time0 = time.time() - time0
    print(f"耗时：{time0}")
    result = {
        "total_entries": 0,
        "data": []
    }

    print('结果如下：')
    entry_count = 0

    for i in range(count):
        if cara[i] != 0:
            entry = {
                "type": "角色",
                "id": i + 1,
                "percentage": f'{caraRT[i] / total_range:.3%}'
            }
            result["data"].append(entry)
            print(f'角色#{i + 1}: {caraRT[i] / total_range:.3%}')
            entry_count += 1

        if ligh[i] != 0:
            entry = {
                "type": "光锥",
                "id": i + 1,
                "percentage": f'{lighRT[i] / total_range:.3%}'
            }
            result["data"].append(entry)
            print(f'光锥#{i + 1}: {lighRT[i] / total_range:.3%}')
            entry_count += 1

    result["total_entries"] = entry_count

    return json.dumps(result, ensure_ascii=False, indent=4)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/update', methods=['POST'])
def update():
    data_dict = json.loads(request.data.decode('utf-8'))
    ticket = int(data_dict['ticket'])
    cL55 = int(data_dict['cL55'])
    lL55 = int(data_dict['lL55'])
    ligh = [int(x) for x in data_dict['ligh'].split(',')]
    cara = [int(x) for x in data_dict['cara'].split(',')]
    count = len(cara)
    result = main(ticket, count, cara, ligh, cL55, lL55)
    return result


if __name__ == '__main__':
    freeze_support()
    server = pywsgi.WSGIServer(('0.0.0.0', 23426), app)
    print("程序启动完成，WebUI地址为 http://127.0.0.1:23426/")
    webbrowser.open('http://127.0.0.1:23426/')
    server.serve_forever()
    tic = 0  # 专票数量
    count = 0
    cara = []  # 预期的角色数量（只）
    ligh = []  # 预期的光锥数量（张）
    # 按照角色[0] 光锥[0] 角色[1] 光锥[1]的顺序交替抽取，0为跳过
