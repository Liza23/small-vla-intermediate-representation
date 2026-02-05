"""
Visualize VLA v1.1 Architecture with Future Prediction
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Color scheme
color_input = '#E8F4F8'
color_encoder = '#B8E6F0'
color_predictor = '#FFD6A5'
color_decoder = '#FFABAB'
color_fusion = '#C7CEEA'
color_flow = '#B5EAD7'
color_output = '#FFDFD3'
color_loss = '#FFB3BA'

# Helper function to draw boxes
def draw_box(x, y, width, height, text, color, fontsize=10, bold=False):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize,
        weight=weight,
        wrap=True
    )

# Helper function to draw arrows
def draw_arrow(x1, y1, x2, y2, label='', color='black', style='->', linewidth=2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        mutation_scale=20
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Title
ax.text(10, 13.5, 'VLA v1.1 Architecture: Future Prediction + Action Generation',
        ha='center', fontsize=18, weight='bold')

# ==================== INPUTS ====================
y_input = 11.5
draw_box(0.5, y_input, 1.5, 0.8, 'Image (t)\n[3,224,224]', color_input, fontsize=9)
draw_box(2.5, y_input, 1.5, 0.8, 'Image (t+1)\n[3,224,224]', color_input, fontsize=9)
draw_box(4.5, y_input, 1.5, 0.8, 'Text\n"pick bowl..."', color_input, fontsize=9)
draw_box(6.5, y_input, 1.3, 0.8, 'Proprio\n[7]', color_input, fontsize=9)
draw_box(8.3, y_input, 1.3, 0.8, 'EE Pose\n[7]', color_input, fontsize=9)
draw_box(18, y_input, 1.5, 0.8, 'Action GT\n[7]', color_input, fontsize=9)

# ==================== ENCODERS ====================
y_encoder = 9.5

# Vision encoder for current image
draw_box(0.5, y_encoder, 1.5, 1.2, 'CLIP Vision\nEncoder\n(frozen)', color_encoder, fontsize=8)
draw_arrow(1.25, y_input, 1.25, y_encoder + 1.2)

# Vision encoder for future image (GT)
draw_box(2.5, y_encoder, 1.5, 1.2, 'CLIP Vision\nEncoder\n(frozen)', color_encoder, fontsize=8)
draw_arrow(3.25, y_input, 3.25, y_encoder + 1.2)
ax.text(3.25, y_encoder - 0.3, 'GT Target', ha='center', fontsize=7, style='italic')

# Text encoder
draw_box(4.5, y_encoder, 1.5, 1.2, 'CLIP Text\nEncoder\n(frozen)', color_encoder, fontsize=8)
draw_arrow(5.25, y_input, 5.25, y_encoder + 1.2)

# Proprio encoder
draw_box(6.5, y_encoder, 1.3, 1.2, 'Proprio\nMLP', color_encoder, fontsize=8)
draw_arrow(7.15, y_input, 7.15, y_encoder + 1.2)

# EE encoder
draw_box(8.3, y_encoder, 1.3, 1.2, 'EE\nMLP', color_encoder, fontsize=8)
draw_arrow(9, y_input, 9, y_encoder + 1.2)

# ==================== EMBEDDINGS ====================
y_embed = 7.8
draw_box(0.5, y_embed, 1.5, 0.6, 'Vision Embed\n[512]', '#D4E8F0', fontsize=8)
draw_arrow(1.25, y_encoder, 1.25, y_embed + 0.6)

draw_box(2.5, y_embed, 1.5, 0.6, 'Target Latent\n[768]', '#FFE5E5', fontsize=8)
draw_arrow(3.25, y_encoder, 3.25, y_embed + 0.6)

draw_box(4.5, y_embed, 1.5, 0.6, 'Text Embed\n[512]', '#D4E8F0', fontsize=8)
draw_arrow(5.25, y_encoder, 5.25, y_embed + 0.6)

draw_box(6.5, y_embed, 1.3, 0.6, 'Proprio\n[512]', '#D4E8F0', fontsize=8)
draw_arrow(7.15, y_encoder, 7.15, y_embed + 0.6)

draw_box(8.3, y_embed, 1.3, 0.6, 'EE\n[512]', '#D4E8F0', fontsize=8)
draw_arrow(9, y_encoder, 9, y_embed + 0.6)

# ==================== FUTURE PREDICTOR (NEW) ====================
y_predictor = 6
draw_box(11, y_predictor, 3, 1.2,
         'Future Latent Predictor\n3-layer MLP [2048→1024→1024→768]\n✨ NEW in v1.1',
         color_predictor, fontsize=9, bold=True)

# Arrows to predictor (concat all embeddings except target)
for x in [1.25, 5.25, 7.15, 9]:
    draw_arrow(x, y_embed, 11, y_predictor + 0.6, '', 'blue', '->', 1.5)

ax.text(9.5, y_predictor + 1.5, 'Concat [2048]', ha='center', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

# Predicted latent output
draw_box(11, y_predictor - 1.2, 3, 0.6, 'Predicted Future Latent\n[768]', color_predictor, fontsize=9, bold=True)
draw_arrow(12.5, y_predictor, 12.5, y_predictor - 0.6)

# ==================== LOSS: FUTURE PREDICTION ====================
draw_arrow(3.25, y_embed, 15.5, y_predictor + 0.2, '', 'red', '->', 1.5)
draw_arrow(12.5, y_predictor - 1.2, 15.5, y_predictor - 0.5, '', 'orange', '->', 1.5)

draw_box(15, y_predictor - 0.7, 2.5, 1.4,
         'Loss: Future\nMSE(pred, target)\nin latent space',
         color_loss, fontsize=9)

# ==================== FUTURE DECODER (NEW) ====================
y_decoder = 3.5
draw_box(11, y_decoder, 3, 1,
         'Future Decoder\n4-layer MLP [768→4096]\n→ [3,64,64]',
         color_decoder, fontsize=9)

draw_arrow(12.5, y_predictor - 1.2, 12.5, y_decoder + 1)

# Decoded image
draw_box(11, y_decoder - 1, 3, 0.6, 'Decoded Future Image\n[3,64,64]', color_decoder, fontsize=8)
draw_arrow(12.5, y_decoder, 12.5, y_decoder - 0.4)

# Decoder loss
draw_box(15, y_decoder - 0.5, 2.5, 1,
         'Loss: Decoder\nMSE(decoded,\ntarget_64x64)',
         color_loss, fontsize=9)

draw_arrow(3.25, y_embed, 16.25, y_decoder + 0.5, '', 'red', '->', 1)
draw_arrow(12.5, y_decoder - 1, 15, y_decoder, '', 'orange', '->', 1)

ax.text(16.25, y_decoder - 1.2, '(visualization)', ha='center', fontsize=7, style='italic')

# ==================== FUSION (MODIFIED) ====================
y_fusion = 6
draw_box(5, y_fusion - 2, 3.5, 1.2,
         'Multimodal Fusion\n5 inputs [2560] → [512]\nIncludes future! 🎯',
         color_fusion, fontsize=9, bold=True)

# Project future to hidden dim first
draw_box(11, y_fusion - 1.5, 3, 0.5, 'Project: [768]→[512]', '#E0E0F0', fontsize=7)
draw_arrow(12.5, y_predictor - 1.2, 12.5, y_fusion - 1)
draw_arrow(12.5, y_fusion - 1.5, 8.5, y_fusion - 1.4, '', 'purple', '->', 2)

# Other embeddings to fusion
for x in [1.25, 5.25, 7.15, 9]:
    draw_arrow(x, y_embed, 6.75, y_fusion - 0.8, '', 'blue', '->', 1.5)

# Condition output
draw_box(5, y_fusion - 3.5, 3.5, 0.7, 'Condition Vector [512]', color_fusion, fontsize=9)
draw_arrow(6.75, y_fusion - 2, 6.75, y_fusion - 2.8)

# ==================== RECTIFIED FLOW ====================
y_flow = 0.8
draw_box(5, y_flow, 3.5, 1.2,
         'Rectified Flow\n4-layer MLP\nDenoising Network',
         color_flow, fontsize=9)

# Condition to flow
draw_arrow(6.75, y_fusion - 3.5, 6.75, y_flow + 1.2)

# Action GT to flow
draw_arrow(18.75, y_input, 18.75, y_flow + 0.6, '', 'red', '->', 1.5)
draw_arrow(18.75, y_flow + 0.6, 8.5, y_flow + 0.6, '', 'red', '->', 1.5)

# Action prediction output
draw_box(5, y_flow - 1.2, 3.5, 0.7, 'Predicted Action [7]', color_output, fontsize=9, bold=True)
draw_arrow(6.75, y_flow, 6.75, y_flow - 0.5)

# ==================== LOSS: ACTION ====================
draw_box(15, y_flow - 0.5, 2.5, 1.4,
         'Loss: Action\nFlow Matching\n(Rectified Flow)',
         color_loss, fontsize=9)

draw_arrow(8.5, y_flow + 0.6, 15, y_flow + 0.4, '', 'red', '->', 1.5)
draw_arrow(8.5, y_flow - 0.5, 15, y_flow - 0.2, '', 'orange', '->', 1.5)

# ==================== TOTAL LOSS ====================
draw_box(15, 1.5, 2.5, 1.5,
         'Total Loss\nloss_future +\nloss_action +\n0.1×loss_decoder',
         '#FF6B6B', fontsize=10, bold=True)

# Arrows from individual losses to total
for y in [y_predictor, y_decoder, y_flow]:
    draw_arrow(16.25, y, 16.25, 3, '', 'darkred', '->', 2)

# ==================== LEGEND ====================
legend_y = 11
legend_elements = [
    ('Input Data', color_input),
    ('Frozen Encoder', color_encoder),
    ('Future Predictor (NEW)', color_predictor),
    ('Future Decoder (NEW)', color_decoder),
    ('Fusion Layer', color_fusion),
    ('Flow Network', color_flow),
    ('Output', color_output),
    ('Loss', color_loss),
]

legend_x = 10.5
for i, (label, color) in enumerate(legend_elements):
    draw_box(legend_x + (i % 4) * 2.3, legend_y - (i // 4) * 0.6, 0.4, 0.3, '', color, fontsize=7)
    ax.text(legend_x + (i % 4) * 2.3 + 0.6, legend_y - (i // 4) * 0.6 + 0.15,
            label, fontsize=7, va='center')

# ==================== KEY INNOVATIONS ====================
ax.text(10, 0.3,
        '🌟 Key Innovation: Model learns to predict future visual state in CLIP latent space, then uses this prediction to guide action generation',
        ha='center', fontsize=10, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD', edgecolor='black', linewidth=2))

# ==================== PARAMETER COUNTS ====================
param_text = '''
Parameter Breakdown:
• CLIP Vision: 86M (frozen) ❄️
• CLIP Text: 63M (frozen) ❄️
• Proprio/EE MLPs: 0.3M
• Future Predictor: 3.2M ✨
• Future Decoder: 4.1M ✨
• Fusion: 1.3M
• Flow Network: 2.1M
━━━━━━━━━━━━━━━━━━━━
Total: 160M (11M trainable)
'''

ax.text(0.5, 5, param_text, fontsize=8, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F8FF', edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig('vla_v1_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Architecture diagram saved: vla_v1_architecture.png")

# Also create a simplified comparison diagram
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# V0 Architecture
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('V0: Direct Action Prediction', fontsize=14, weight='bold', pad=20)

# V0 flow
draw_box_simple = lambda ax, x, y, w, h, text, color: ax.add_patch(
    FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                   edgecolor='black', facecolor=color, linewidth=2)
) or ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, weight='bold')

draw_box_simple(ax1, 1, 8, 8, 1, 'Inputs: Image(t) + Text + Proprio + EE', color_input)
draw_box_simple(ax1, 1, 6, 8, 1.3, 'CLIP Encoders + MLPs\n[frozen vision/text]', color_encoder)
draw_box_simple(ax1, 1, 4, 8, 1.3, 'Fusion: Concat all → [512]', color_fusion)
draw_box_simple(ax1, 1, 2, 8, 1.3, 'Rectified Flow → Action [7]', color_flow)
draw_box_simple(ax1, 1, 0.2, 8, 1, 'Loss: Action only', color_loss)

# Arrows
for y in [8, 6, 4, 2]:
    ax1.arrow(5, y, 0, -0.6, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)

# V1 Architecture
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('V1.1: Future Prediction + Action', fontsize=14, weight='bold', pad=20)

draw_box_simple(ax2, 1, 8, 8, 1, 'Inputs: Image(t) + Image(t+1) + Text + Proprio + EE', color_input)
draw_box_simple(ax2, 1, 6, 8, 1.3, 'CLIP Encoders + MLPs', color_encoder)
draw_box_simple(ax2, 1, 4, 3.5, 1.3, 'Future Predictor\n[768] ✨', color_predictor)
draw_box_simple(ax2, 5.5, 4, 3.5, 1.3, 'Decoder [64×64]\n(viz) 🎨', color_decoder)
draw_box_simple(ax2, 1, 2, 8, 1.3, 'Fusion: +Predicted Future → Flow → Action', color_flow)
draw_box_simple(ax2, 1, 0.2, 8, 1, 'Loss: Future + Action + Decoder', color_loss)

# Arrows
ax2.arrow(3, 6, 0, -0.6, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)
ax2.arrow(7, 6, 0, -0.6, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)
ax2.arrow(2.75, 5.3, 1.75, -1.3, head_width=0.3, head_length=0.2, fc='purple', ec='purple', linewidth=2)
ax2.arrow(5, 4, 0, -0.6, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)
ax2.arrow(5, 2, 0, -0.6, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)

plt.tight_layout()
plt.savefig('vla_v0_vs_v1_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Comparison diagram saved: vla_v0_vs_v1_comparison.png")

plt.show()
