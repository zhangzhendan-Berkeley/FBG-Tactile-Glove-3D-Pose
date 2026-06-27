import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14

fig, ax = plt.subplots(figsize=(7, 6))

# 数据点
ax.scatter(0.916, 1.209, c='red', marker='*', s=180)
ax.scatter(1.47,  0.8,   c='blue', marker='s', s=100)
ax.scatter(0.1,   2.6,   c='teal', marker='D', s=100)
ax.scatter(5.0,   1.8,   c='orange', marker='^', s=100)
ax.scatter(4.02,  4.16,  c='purple', marker='p', s=100)

# HandCept（半透明灰色）
ax.scatter(5.8, 3.0, c='gray', marker='X', s=100, alpha=0.6, edgecolors='none')

# DeepFisheye（显示位置压缩后）
ax.scatter(6.2, 0.3, c='gray', marker='*', s=100, alpha=0.6, edgecolors='none')

# 直接标注：文字 + 箭头
ax.annotate('Ours',
            xy=(0.916, 1.209), xytext=(1.35, 1.55),
            fontsize=16,
            arrowprops=dict(arrowstyle='->', lw=1.0))

ax.annotate('ASTRA Glove',
            xy=(1.47, 0.8), xytext=(2.05, 0.45),
            fontsize=16,
            arrowprops=dict(arrowstyle='->', lw=1.0))

ax.annotate('OneTip',
            xy=(0.1, 2.6), xytext=(0.8, 3.0),
            fontsize=16,
            arrowprops=dict(arrowstyle='->', lw=1.0))

ax.annotate('Kortier et al.',
            xy=(5.0, 1.8), xytext=(4.15, 1.1),
            fontsize=16,
            arrowprops=dict(arrowstyle='->', lw=1.0))

ax.annotate('Park et al.',
            xy=(4.02, 4.16), xytext=(3.0, 4.7),
            fontsize=16,
            arrowprops=dict(arrowstyle='->', lw=1.0))

ax.annotate('HandCept',
            xy=(5.8, 3.0), xytext=(4.7, 3.5),
            fontsize=16,
            arrowprops=dict(arrowstyle='->', lw=1.0))

ax.annotate('DeepFisheye$^*$',
            xy=(6.2, 0.3), xytext=(4.7, 0.6),
            fontsize=16,
            arrowprops=dict(arrowstyle='->', lw=1.0))

ax.set_xlim(0, 6.5)
ax.set_ylim(0, 5.5)
ax.set_xlabel('Position Error (mm)')
ax.set_ylabel('Rotation / Joint Angle Error (°)')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 不再显示图例
# legend = ax.legend(...)

# 添加脚注说明 DeepFisheye 真实误差
plt.figtext(0.5, 0.02, '* DeepFisheye actual position error is 20 mm.',
            ha='center', fontsize=16, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig('sota_scatter.pdf', format='pdf', dpi=800, bbox_inches='tight')
plt.show()