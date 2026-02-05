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
non_cohnum = 8 # ノンコヒーレント積分時間(ms)
f_doppler_candidate = np.arange(-5000, 5000, 500) # ドップラー周波数の探索範囲
#f_doppler_candidate = np.array([-2000]) # ドップラー周波数の探索範囲
LOAD_LENGTH = int(fs * 50e-3)
PRN = 17
FILE = "./2026-01-30_TMU/2026-01-30T20-03-00_I1_16Msps_1Gsamples.bin"
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

def readdata(f_name, start, num_sample):
    # ベースバンドデータの読み込み
    I = np.fromfile(f_name, dtype=np.int8, count = num_sample, offset = start).astype(np.uint8)

    samples = len(I)
    print("Loaded {0:.1f} ms".format(samples/fs*1000))

    i_tmp = np.zeros(samples).astype(np.int8)
    q_tmp = np.zeros(samples).astype(np.int8)

    i_tmp[0::4] = I[0::4]
    i_tmp[2::4] = -1*I[2::4]

    q_tmp[1::4] = I[1::4]
    q_tmp[3::4] = -1*I[3::4]

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

samples, i, q = readdata(FILE, 0, LOAD_LENGTH)

# C/Aコードを4092点まで引き伸ばす
chip_index = (np.floor(np.arange(num_coherent_data_sample) * fcw) % code_len).astype(int)

# 結果の格納用変数の作成。ドップラー周波数とコード遅延の2軸あるので、2次元の配列にする。
corr_map = np.zeros((len(f_doppler_candidate), code_len))

# 力技処理の部分。ドップラーシフトとコード遅延を総当りする。
for fi, f_doppler in enumerate(f_doppler_candidate): # f_dopplerがドップラー周波数
    print(f"Search Doppler frequency {f_doppler} Hz")
    n = np.arange(samples)
    N = int(f_doppler/fb*np.iinfo(np.uint32).max)

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
            #if (f_doppler == -2000 and code_delay == 163):
            #    print("i_corr: {}".format(i_corr))
            #    print("q_corr: {}".format(q_corr))
            #    print("Power: {}".format(power_sum))
            #    print()
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

print()
print()

LOAD_LENGTH = int(fs*500e-3)
TOTAL_LENGTH = int(fs*3000e-3)
TOTAL_SAMPLES = TOTAL_LENGTH//4

DP_NCO_FULL = np.iinfo(np.uint32).max
doppler_nco = 0
dp_ack_omega = int(f_doppler_candidate[max_fd_index]/fb*DP_NCO_FULL)
doppler_omega = dp_ack_omega

localcode = np.roll(prn31, max_code_index)
CODE_NCO_FULL = np.iinfo(np.uint32).max
code_nco_el = CODE_NCO_FULL//2
code_nco_punctual = 0

CODE_NCO_INIT = int(fcw*(CODE_NCO_FULL+1))
code_nco_omega = 0
code_error = 0
code_error_prev = 0

code_phase_early = 1022
code_phase_punctual = 0
code_phase_late = 0

coherent_data_counter = 0
integrator_i_punctual = 0
integrator_q_punctual = 0
integrator_i_early = 0
integrator_q_early = 0
integrator_i_late = 0
integrator_q_late = 0

track_punctual_i = np.zeros(TOTAL_SAMPLES//num_coherent_data_sample + 1)
track_punctual_q = np.zeros(TOTAL_SAMPLES//num_coherent_data_sample + 1)
#track_early_i = np.zeros(samples//num_coherent_data_sample + 1)
#track_early_q = np.zeros(samples//num_coherent_data_sample + 1)
#track_late_i = np.zeros(samples//num_coherent_data_sample + 1)
#track_late_q = np.zeros(samples//num_coherent_data_sample + 1)

code_errors = np.zeros(TOTAL_SAMPLES//num_coherent_data_sample+1)
dp_errors = np.zeros(TOTAL_SAMPLES//num_coherent_data_sample+1)

code_nco_omegas = np.zeros(TOTAL_SAMPLES//num_coherent_data_sample+1)
dp_nco_omegas = np.zeros(TOTAL_SAMPLES//num_coherent_data_sample+1)

non_cohs = np.zeros(TOTAL_SAMPLES//(num_coherent_data_sample * non_cohnum))
non_coh_index = 0

sample_counter = 0
index_counter = 0

incoh_counter = 0
incoh_integ = 0

mode_count = 0

carrier_i = 0;
carrier_q = 0;
i_mixed = 0;
q_mixed = 0;

prev_in = 0
prev_qn = 0

dp_error = 0
dp_error_prev = 0
dp_fll_error = 0

df_fll = 0
df_pll = 0
fll_sigma = 0
pll_sigma = 0

costas_sigma = 0

ki_100hz = 6.3e-2
ki_80hz = 4.03e-2
ki_60hz = 2.27e-2
ki_50hz = 1.58e-2
ki_40hz = 1.01e-2
ki_30hz = 5.69e-3
ki_20hz = 1.42e-3
ki_10hz = 6.30e-4
ki_5hz = 1.58e-4
ki_2hz = 2.52e-5
ki_1hz = 6.3e-6
ki_05hz = 1.58e-6

kp_100hz = 0.355
kp_80hz = 0.284
kp_60hz = 0.213
kp_50hz = 0.178
kp_40hz = 0.142
kp_30hz = 0.1066
kp_20hz = 0.0533
kp_10hz = 0.0355
kp_5hz = 0.0178
kp_2hz = 0.00710
kp_1hz = 0.00355
kp_05hz = 0.00178
#ki_WB = 0.01
#kp_WB = 0.08
# 20Hz by gemini
#ki_FLL_WB = 0.0158
#kp_FLL_WB = 0.178

# 40Hz by ChatGPT
#ki_WB = 0.0101
#kp_WB = 0.142

#ki_NB = 0.01#0.0016
#kp_NB = 0.08#0.032
# 10Hz by gemini
#ki_NB = 0.00141
#kp_NB = 0.052
ki_FLL_WB = ki_20hz
kp_FLL_WB = kp_20hz
ki_FLL_MB = ki_10hz
kp_FLL_MB = kp_10hz
ki_FLL_NB = ki_5hz
kp_FLL_NB = kp_5hz

k_fll = 100000
k_fll_conf = 1

ep_fll = 20

#kp_fll = 0.02

code_sigma = 0
ki_DLL_WB = ki_5hz
kp_DLL_WB = kp_5hz
ki_DLL_MB = ki_2hz
kp_DLL_MB = kp_2hz
ki_DLL_NB = ki_2hz
kp_DLL_NB = kp_2hz

k_code = -15000000
ep_code = 20

ki_PLL_WB = ki_20hz
kp_PLL_WB = kp_20hz
ki_PLL_MB = ki_20hz
kp_PLL_MB = kp_20hz
ki_PLL_NB = ki_20hz
kp_PLL_NB = kp_20hz

k_pll = 0#.001
k_pll_conf = 1
ep_pll = 10

ep = 10

ki_FLL = ki_FLL_WB
kp_FLL = kp_FLL_WB
ki_PLL = ki_PLL_WB
kp_PLL = kp_PLL_WB
ki_DLL = ki_DLL_WB
kp_DLL = kp_DLL_WB

TIME_WB = 500
TIME_MB = 1500
TIME_PLL = 4000
TIME_PHADE = 5000

wb_fll = 1
lock_pll = 0
wb_pll = 1
mode_counter = 0
mode = "Wideband"

alpha = 1
phade_step = 0.001
df = 0
prev_df = 0
dc = 0
prev_dc = 0
int_df = 0

for num in range(TOTAL_LENGTH//LOAD_LENGTH):
    samples, i, q = readdata(FILE, LOAD_LENGTH*num, LOAD_LENGTH)

    for di, dq in zip(i, q):
        carrier_i = cos(doppler_nco >> 30)
        carrier_q = sin(doppler_nco >> 30)

        i_mixed = di * carrier_i
        q_mixed = dq * carrier_q

        i_corr_early = i_mixed * localcode[code_phase_early]
        i_corr_punctual = i_mixed * localcode[code_phase_punctual]
        i_corr_late = i_mixed * localcode[code_phase_late]

        q_corr_early = q_mixed * localcode[code_phase_early]
        q_corr_punctual = q_mixed * localcode[code_phase_punctual]
        q_corr_late = q_mixed * localcode[code_phase_late]

        integrator_i_early += i_corr_early
        integrator_i_punctual += i_corr_punctual
        integrator_i_late += i_corr_late

        integrator_q_early += q_corr_early
        integrator_q_punctual += q_corr_punctual
        integrator_q_late += q_corr_late

        doppler_nco = (doppler_nco + doppler_omega) % DP_NCO_FULL
        code_nco_punctual = (code_nco_punctual + code_nco_omega)
        code_nco_el = (code_nco_el+code_nco_omega)

        if CODE_NCO_FULL < code_nco_punctual:
            code_nco_punctual = code_nco_punctual % CODE_NCO_FULL

            if code_phase_punctual < 1022:
                code_phase_punctual += 1
            else:
                code_phase_punctual = 0


        if CODE_NCO_FULL < code_nco_el:
            code_nco_el = code_nco_el % CODE_NCO_FULL

            if code_phase_early < 1022:
                code_phase_early += 1
            else:
                code_phase_early = 0

            if code_phase_late < 1022:
                code_phase_late += 1
            else:
                code_phase_late = 0

        coherent_data_counter += 1

        if coherent_data_counter > (num_coherent_data_sample - 1):
            #track_late_i[index_counter] = integrator_i_late
            #track_late_q[index_counter] = integrator_q_late
            #track_early_i[index_counter] = integrator_i_early
            #track_early_q[index_counter] = integrator_q_early
            track_punctual_i[index_counter] = integrator_i_punctual
            track_punctual_q[index_counter] = integrator_q_punctual

            #print("i_corr: {}".format(integrator_i_punctual))
            #print("q_corr: {}".format(integrator_q_punctual))
            #print()

            if mode_counter < TIME_WB:
                ki_FLL = ki_FLL_WB
                kp_FLL = kp_FLL_WB
                ki_DLL = ki_DLL_WB
                kp_DLL = kp_DLL_WB
                mode = "Wideband"
            elif mode_counter >= TIME_WB and mode_counter < TIME_MB:
                ki_FLL = ki_FLL_MB
                kp_FLL = kp_FLL_MB
                ki_DLL = ki_DLL_MB
                kp_DLL = kp_DLL_MB
                mode = "Midband"
            elif mode_counter >= TIME_MB and mode_counter < TIME_PLL:
                ki_FLL = ki_FLL_NB
                kp_FLL = kp_FLL_NB
                ki_DLL = ki_DLL_NB
                kp_DLL = kp_DLL_NB
                mode = "Narrowband"
            else:
                ki_PLL = ki_PLL_NB
                kp_PLL = kp_PLL_NB
                mode = "PLL"

            if alpha < 0.1:
                ki_PLL = ki_PLL_WB
                kp_PLL = kp_PLL_WB
            elif alpha < 0.3:
                ki_PLL = ki_PLL_MB
                kp_PLL = kp_PLL_MB


            # FLL ERROR
            cross = integrator_i_punctual*prev_qn - integrator_q_punctual*prev_in
            dot = integrator_i_punctual*prev_in + integrator_q_punctual*prev_qn
            p = integrator_i_punctual**2 + integrator_q_punctual**2
            p_prev = prev_in**2 + prev_qn**2
            dp_fll_error = cross/(p * p_prev + ep_fll)/(2.0*np.pi)*DP_NCO_FULL*coherent_time*k_fll
            fll_sigma = fll_sigma + ki_FLL*dp_fll_error
            df_fll = int((fll_sigma + kp_FLL*dp_fll_error))

            # PLL error
            #dot = integrator_i_punctual**2 + integrator_q_punctual**2
            #cross = integrator_i_punctual * integrator_q_punctual
            dp_pll_error = integrator_q_punctual * np.sign(integrator_i_punctual)*k_pll #integrator_q_punctual * integrator_i_punctual * k_pll #integrator_q_punctual/(dot+ep_pll)*k_pll
            costas_sigma = costas_sigma + ki_PLL*dp_pll_error
            df_pll = int((costas_sigma + kp_PLL*dp_pll_error)/(2.0*np.pi)*DP_NCO_FULL*coherent_time*k_pll_conf)

            if  mode_counter >= TIME_PLL and mode_counter < TIME_PHADE:
                alpha -= phade_step

            df = df_fll + (1-alpha)*df_pll
            #int_df = int_df + dp_error * ki
            #df = int(int_df + (kp*dp_error)) #int(alpha*df_fll + (1-alpha)*df_pll)
            diff = df-prev_df
            if np.abs(diff) > 100000:
                if diff < 0:
                    df = prev_df - 100000
                else:
                    df = prev_df + 100000

            doppler_omega = int(dp_ack_omega - df)
            
            E = np.sqrt(integrator_i_early**2 + integrator_q_early**2)
            L = np.sqrt(integrator_i_late**2 + integrator_q_late**2)
            #code_error = ((integrator_i_late - integrator_i_early)*integrator_i_punctual + (integrator_q_late - integrator_q_early)*integrator_q_punctual)
            code_error = (E-L)/(E+L+ep_code)*k_code
            code_sigma = code_sigma + ki_DLL*code_error
            dc = int((code_sigma + kp_DLL*code_error))
            diff_dc = dc - prev_dc
            if np.abs(diff_dc) > 1000000:
                if diff_dc < 0:
                    dc = prev_dc - 1000000
                else:
                    dc = prev_dc + 1000000

            code_nco_omega = CODE_NCO_INIT + dc

            code_error_prev = code_error
            prev_in = integrator_i_punctual
            prev_qn = integrator_q_punctual
            prev_df = df
            prev_dc = dc

            dp_errors[index_counter] = dp_pll_error #dp_error
            dp_nco_omegas[index_counter] = doppler_omega
            code_errors[index_counter] = code_error
            code_nco_omegas[index_counter] = code_nco_omega
            incoh_integ += integrator_i_punctual*integrator_i_punctual + integrator_q_punctual*integrator_q_punctual
            #print("Incoh integ: {}".format(incoh_integ))
            incoh_counter += 1
            mode_counter += 1

            if incoh_counter > non_cohnum:
                print("Elapsed time: {} ms".format(mode_counter))
                print("Incoh integ: {}".format(incoh_integ))
                #print("I early: {}".format(integrator_i_early))
                #print("I late: {}".format(integrator_i_late))
                #print("Q early: {}".format(integrator_q_early))
                #print("Q late: {}".format(integrator_q_late))
                #print("I punctual: {}".format(integrator_i_punctual))
                #print("Q punctual: {}".format(integrator_q_punctual))
                print("CODE ERR: {}".format(code_error))
                print("CODE omega: {}".format(code_nco_omega))
                #print("Lock mode: {}".format("FLL" if lock_pll == 0 else "Costas PLL"))
                #print("FLL mode: {}".format("Narrow" if wb_fll == 0 else "Wide"))
                #print("PLL mode: {}".format("Narrow" if wb_pll == 0 else "Wide"))
                print("Mode: {}".format(mode))
                print("ALPHA: {}".format(alpha))
                print("DP ERR: {}".format(dp_error))
                print("DP_FLL_ERR: {}".format(dp_fll_error))
                print("DP_PLL_ERR: {}".format(dp_pll_error))
                print("DF: {}".format(df))
                print("DP omega: {}".format(doppler_omega))
                print()

                #if incoh_integ > 3e6:
                    #wb_fll = 0
                
                non_cohs[non_coh_index] = incoh_integ
                non_coh_index += 1
                incoh_counter = 0
                incoh_integ = 0

            integrator_i_early = 0
            integrator_q_early = 0
            integrator_i_punctual = 0
            integrator_q_punctual = 0
            integrator_i_late = 0
            integrator_q_late = 0

            coherent_data_counter = 0
            index_counter += 1


fig = plt.figure(figsize=(10,14))
ax = fig.add_subplot(711)
ax.plot(track_punctual_i)
ax.plot(track_punctual_q)
ax.grid(ls='--')

ac = fig.add_subplot(712)
ac.plot(1,1)
ac.set_xlim((0,10))
ac.set_ylim((0,10))
ac.text(0.02, 0.98,
        f"PRN: {PRN}", 
        transform=ac.transAxes,
        ha = 'left',
        va = 'top') 
ac.text(0.02, 0.88,
        f"Wide: DLL_p={kp_DLL_WB}, DLL_i={ki_DLL_NB}, FLL_p={kp_FLL_WB}, FLL_i={ki_FLL_WB}",
        transform=ac.transAxes,
        ha = 'left',
        va = 'top') 
ac.text(0.02, 0.78,
        f"Mid: DLL_p={kp_DLL_MB}, DLL_i={ki_DLL_MB}, FLL_p={kp_FLL_MB}, FLL_i={ki_FLL_MB}",
        transform=ac.transAxes,
        ha = 'left',
        va = 'top') 
ac.text(0.02, 0.68,
        f"Narrow: DLL_p={kp_DLL_NB}, DLL_i={ki_DLL_NB}, FLL_p={kp_FLL_NB}, FLL_i={ki_FLL_NB}",
        transform=ac.transAxes,
        ha = 'left',
        va = 'top') 
ac.text(0.02, 0.58,
        f"PLL: PLL_p={kp_PLL_WB}, PLL_i={ki_PLL_WB}", 
        transform=ac.transAxes,
        ha = 'left',
        va = 'top') 
ac.text(0.02, 0.48,
        f"Gain: k_code = {k_code}, k_fll={k_fll}, k_pll={k_pll}",
        transform=ac.transAxes,
        ha = 'left',
        va = 'top') 
ac.text(0.02, 0.38,
        f"Conversion gain: k_fll_conv={k_fll_conf}, k_pll_conf={k_pll_conf}",
        transform=ac.transAxes,
        ha = 'left',
        va = 'top')
ac.text(0.02, 0.28,
        f"WIDE: {TIME_WB} ms, MID: {TIME_MB}, PLL START: {TIME_PLL}",
        transform=ac.transAxes,
        ha = 'left',
        va = 'top') 
ac.text(0.02, 0.18,
        f"File: {FILE}",
        transform=ac.transAxes,
        ha = 'left',
        va = 'top') 
aa = fig.add_subplot(713, sharex=ax)
aa.set_title("Code error")
aa.plot(code_errors)
aa.grid(ls='--')
ad = fig.add_subplot(714, sharex=ax)
ad.set_title("Code omega")
ad.plot(code_nco_omegas[:-2])
ad.grid(ls='--')

ab = fig.add_subplot(715, sharex=ax)
ab.set_title("Doppler Error")
ab.plot(dp_errors)
ab.grid(ls='--')

ab = fig.add_subplot(716, sharex=ax)
ab.set_title("Doppler omega")
ab.plot(dp_nco_omegas[:-2])
ab.grid(ls='--')


ay = fig.add_subplot(717)
ay.plot(track_punctual_i[1500:], track_punctual_q[1500:], '.')
#ay.set_ylim((-1000,1000))
#ay.set_xlim((-1000,1000))
ay.set_xlabel('I')
ay.set_ylabel('Q')
ay.set_aspect("equal")
ay.grid(ls='--')

fig.tight_layout()
plt.show()
