import numpy as np
import matplotlib.pyplot as plt

# 定数とか
fs = 16.368e6 # ベースバンド信号のサンプリングレート(Hz)
fIF = 4.092e6 # IF 周波数 (Hz)
chip_rate = 1.023e6 # C/Aコードのチップレート(Hz)
code_len = 1023 # C/Aコード1周期の長さ
fcw = chip_rate / fs # ベースバンド1サンプルに対するC/Aコードの更新速度
coherent_time = 1e-3 # 相互相関する時間 (s)
num_coherent_data_sample = int(fs*coherent_time) # 相互相関するサンプル数→4092点
non_cohnum = 16 # ノンコヒーレント積分時間(ms)
f_doppler_candidate = np.arange(-5000, 5000, 500) # ドップラー周波数の探索範囲
PRN = 29
FILE = "./2025-12-32T03-32-00_L1.bin"

print("FILE: {}".format(FILE))
print("Search PRN: {}".format(PRN))
print("Number of coherent data samples: {}".format(num_coherent_data_sample))


#
def prn_taps(prn):
    """
    GPS L1 C/A
    PRN番号 → G2 LFSR タップ位置（0オリジン）
    戻り値: [tap1, tap2]
    """

    g2_tap_table_0based = {
         1:  [1, 5],
         2:  [2, 6],
         3:  [3, 7],
         4:  [4, 8],
         5:  [0, 8],
         6:  [1, 9],
         7:  [0, 7],
         8:  [1, 8],
         9:  [2, 9],
        10:  [1, 2],
        11:  [2, 3],
        12:  [4, 5],
        13:  [5, 6],
        14:  [6, 7],
        15:  [7, 8],
        16:  [8, 9],
        17:  [0, 3],
        18:  [1, 4],
        19:  [2, 5],
        20:  [3, 6],
        21:  [4, 7],
        22:  [5, 8],
        23:  [0, 2],
        24:  [3, 5],
        25:  [4, 6],
        26:  [5, 7],
        27:  [6, 8],
        28:  [7, 9],
        29:  [0, 5],
        30:  [1, 6],
        31:  [2, 7],
        32:  [3, 8],
        33:  [4, 9],
        34:  [3, 9],
        35:  [0, 6],
        36:  [1, 7],
        37:  [3, 9],
    }

    if prn not in g2_tap_table_0based:
        raise ValueError("PRN must be between 1 and 37")

    return g2_tap_table_0based[prn]


# C/Aコードの生成
def shift(g1, g2):
  fb1 = g1[2]^g1[9]
  fb2 = g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]

  g1 = np.roll(g1, 1)
  g1[0] = fb1
  g2 = np.roll(g2, 1)
  g2[0] = fb2

  return (g1, g2)

def cacode(g1, g2, sat):
  return g1[9]^g2[sat[0]]^g2[sat[1]]

g1 = np.ones(10).astype('int8')
g2 = np.ones(10).astype('int8')

prn31 = np.zeros(1023)

for n in range(1023):
  prn31[n] = cacode(g1, g2, prn_taps(PRN))
  g1, g2 = shift(g1, g2)

# C/Aコードを16368点まで引き伸ばす
chip_index = (np.floor(np.arange(num_coherent_data_sample) * fcw) % code_len).astype(int)

# ベースバンドデータの読み込み
I = np.fromfile(FILE, dtype=np.int8, count = 16368*20)
#i = 1 - 2*i

samples = len(I)

i = np.zeros(samples)
q = np.zeros(samples)

i[0::4] = I[0::4]
i[2::4] = -1*I[2::4]

q[1::4] = -1*I[1::4]
q[3::4] = I[3::4]

print("Loaded {0:.1f} ms".format(samples/fs*1000))

# 結果の格納用変数の作成。ドップラー周波数とコード遅延の2軸あるので、2次元の配列にする。
corr_map = np.zeros((len(f_doppler_candidate), code_len))

# 力技処理の部分。ドップラーシフトとコード遅延を総当りする。
for fi, f_doppler in enumerate(f_doppler_candidate): # f_dopplerがドップラー周波数
  print(f"Search Doppler frequency {f_doppler} Hz")
  n = np.arange(samples)

  # ドップラーシフト補正用キャリア信号生成
  phase = 2*np.pi*f_doppler/fs*n
  carrier_i = np.sign(np.cos(phase)).astype(np.int8)
  carrier_q = np.sign(np.sin(phase)).astype(np.int8)

  # ドップラーシフトの補正
  i_mixed = carrier_i*i
  q_mixed = carrier_q*q

  power_sum = 0
  # コヒーレント積分
  for code_delay in range(code_len):
    local_code = np.roll(prn31, code_delay)[chip_index]

    power_sum = 0.0
    # ノンコヒーレント積分
    for blk in range(non_cohnum):
      start = blk * int(fs * coherent_time)
      stop = start + int(fs * coherent_time)
      # 相関処理
      i_corr = np.sum(i_mixed[start:stop] * local_code)
      q_corr = np.sum(q_mixed[start:stop] * local_code)
      # ノンコヒーレント積分の足し合わせ処理
      power_sum += i_corr*i_corr + q_corr*q_corr
    corr_map[fi, code_delay] = power_sum

# 結果表示部分
max_fd_index, max_code_index = np.unravel_index(np.argmax(corr_map), corr_map.shape)
print(f"Detected peak: Doppler={f_doppler_candidate[max_fd_index]} Hz, Code phase = {max_code_index}, Corr = {corr_map[max_fd_index, max_code_index]}")

fig = plt.figure()
ax = fig.add_subplot(211)
im1 = ax.imshow(corr_map.T, aspect='auto',
                extent = [f_doppler_candidate[0], f_doppler_candidate[-1], 0, 1023],
                origin='lower', cmap = 'inferno')
ax.set_xlabel("Doppler (Hz)")
ax.set_ylabel("Code phase (chips)")
fig.colorbar(im1, label='Correlation')

ay = fig.add_subplot(212)
ay.plot(np.arange(1023), corr_map[max_fd_index, :])
ay.set_xlabel("Code phase (chips)")
ay.set_ylabel("Correlation")
ay.text(0.98, 0.98,
        f"PRN: {PRN}",
        transform=ay.transAxes,
        ha = 'right',
        va = 'top') 
ay.text(0.98, 0.88,
        f"Detected peak: Doppler={f_doppler_candidate[max_fd_index]} Hz",
        transform=ay.transAxes,
        ha = 'right',
        va = 'top') 
ay.text(0.98, 0.78,
        f"Code phase = {max_code_index}",
        transform=ay.transAxes,
        ha = 'right',
        va = 'top') 
ay.text(0.98, 0.68,
        f"Corr = {corr_map[max_fd_index, max_code_index]}",
        transform=ay.transAxes,
        ha = 'right',
        va = 'top') 
fig.tight_layout()
plt.show()


