import numpy as np
import matplotlib.pyplot as plt

# 定数とか
fs = 16.368e6 # IF信号のサンプリングレート(Hz)
fIF = 4.092e6 # IF 周波数 (Hz)
fb = 4.092e6 # ベースバンド信号のサンプリングレート
chip_rate = 1.023e6 # C/Aコードのチップレート(Hz)
code_len = 1023 # C/Aコード1周期の長さ
fcw = chip_rate / fb # ベースバンド1サンプルに対するC/Aコードの更新速度
coherent_time = 1e-3 # 相互相関する時間 (s)
num_coherent_data_sample = int(fb*coherent_time) # 相互相関するサンプル数→4092点
non_cohnum = 16 # ノンコヒーレント積分時間(ms)
f_doppler_candidate = np.arange(-5000, 5000, 500) # ドップラー周波数の探索範囲
LOAD_LENGTH = int(fs * 20e-3)
PRN = 29
INITIAL_CODE_PHASE = 163
INITIAL_DOPPLER_FREQUENCY = -2000
FILE = "./2025-12-32T03-32-00_L1.bin"

FULL_INT32 = np.iinfo(np.int32).max

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


def gen_cacode(prn):
    g1 = np.ones(10).astype(np.uint8)
    g2 = np.ones(10).astype(np.uint8)

    ca_code = np.zeros(1023).astype(np.int8)

    for n in range(1023):
      ca_code[n] = cacode(g1, g2, prn_taps(prn))
      g1, g2 = shift(g1, g2)

    return (1 - 2*ca_code)

def readdata(f_name, num_sample):
    # ベースバンドデータの読み込み
    I = np.fromfile(f_name, dtype=np.int8, count = num_sample).astype(np.uint8)

    samples = len(I)
    print("Loaded {0:.1f} ms".format(samples/fs*1000))

    i_tmp = np.zeros(samples).astype(np.int8)
    q_tmp = np.zeros(samples).astype(np.int8)

    i_tmp[0::4] = I[0::4]
    i_tmp[2::4] = -1*I[2::4]

    q_tmp[1::4] = -1*I[1::4]
    q_tmp[3::4] = I[3::4]

    i = np.sum(i_tmp.reshape((-1, 4)), axis = 1)
    q = np.sum(q_tmp.reshape((-1, 4)), axis = 1)

    samples = samples // 4

    return (samples, i, q)

def cos(param):
    ss = np.array([1,1,-1,-1]).astype(np.int8)
    t = param & 0x3

    return ss[t]

def sin(param):
    ss = np.array([0, 1, 1, -1]).astype(np.int8)
    t = param & 0x3

    return ss[t]

def xor_corr(data_bits, code_bits):
    xor = np.bitwise_xor(data_bits, code_bits)
    return np.sum(1 - 2*xor)

# C/Aコード生成
prn31 = np.array(gen_cacode(PRN))

samples, i, q = readdata(FILE, LOAD_LENGTH)

# C/Aコードを4092点まで引き伸ばす
chip_index = (np.floor(np.arange(num_coherent_data_sample) * fcw) % code_len).astype(int)

# 結果の格納用変数の作成。ドップラー周波数とコード遅延の2軸あるので、2次元の配列にする。
corr_map = np.zeros((len(f_doppler_candidate), code_len))

# 力技処理の部分。ドップラーシフトとコード遅延を総当りする。
for fi, f_doppler in enumerate(f_doppler_candidate): # f_dopplerがドップラー周波数
    print(f"Search Doppler frequency {f_doppler} Hz")
    n = np.arange(samples)
    N = int(f_doppler/fb*np.iinfo(np.uint32).max)
    print(f"DP: {N}")

    # ドップラーシフト補正用キャリア信号生成
    phase = (((N*n)%np.iinfo(np.uint32).max) >> 30).astype(np.uint8)
    carrier_i = cos(phase)
    carrier_q = sin(phase)

    # ドップラーシフトの補正
    i_mixed = carrier_i * i
    q_mixed = carrier_q * q

    power_sum = 0
    # コヒーレント積分
    for code_delay in range(code_len):
        local_code = np.roll(prn31, code_delay)[chip_index]

        power_sum = 0.0
        # ノンコヒーレント積分
        for blk in range(non_cohnum):
            start = blk * num_coherent_data_sample
            stop = start + num_coherent_data_sample
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

#carrier_i = 0;
#carrier_q = 0;
#i_mixed = 0;
#q_mixed = 0;
#
#doppler_nco = 0
#doppler_omega = 3
#
#in0 = 0
#qn0 = 0
#dp_error = 0
#dp_error_prev = 0
#DP_NCO_FULL = 0xffffffff
#CODE_NCO_FULL = 0xffffffff
#
#cacode = np.roll(prn31, INITIAL_CODE_PHASE + 1)
#
#code_nco = 0
#code_error = 0
#code_error_prev = 0
#
#half_late = 0
#
#code_phase_early = 0
#code_phase_punctual = 0
#code_phase_late = 1
#
#coherent_data_counter = 0
#integrator_i_punctual = 0
#integrator_q_punctual = 0
#integrator_i_early = 0
#integrator_q_early = 0
#integrator_i_late = 0
#integrator_q_late = 0
#
#track_punctual_i = np.zeros(samples//num_coherent_data_sample + 1)
#track_punctual_q = np.zeros(samples//num_coherent_data_sample + 1)
#track_early_i = np.zeros(samples//num_coherent_data_sample + 1)
#track_early_q = np.zeros(samples//num_coherent_data_sample + 1)
#track_late_i = np.zeros(samples//num_coherent_data_sample + 1)
#track_late_q = np.zeros(samples//num_coherent_data_sample + 1)
#
#code_errors = np.zeros(samples//num_coherent_data_sample+1)
#dp_error = np.zeros(samples//num_coherent_data_sample+1)
#
#code_nco_omegas = np.zeros(samples//num_coherent_data_sample+1)
#dp_nco_omegas = np.zeros(samples//num_coherent_data_sample+1)
#
#demod_i = np.zeros(samples)
#demod_q = np.zeros(samples)
#
#sample_counter = 0
#index_counter = 0
#
#incoh_counter = 0
#incoh_integ = 0
#
#mode_count = 0
#
#
#for di, dq in zip(i, q):
#    carrier_i = cos(doppler_nco >> 28)
#    carrier_q = sin(doppler_nco >> 28)
#
