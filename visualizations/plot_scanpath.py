import argparse
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

SCALING_FACTOR = 8


def sample_and_plot_scanpaths(
        file_path: str,
        max_words: int,
        num_sample_sp: int = 10,
        letter_height=60,
        img_size=(1400, 800),
        rank: int = 1,
        batch_id: int = 1,
        output_dir: str = 'scanpaths',
):
    """
    Plots the original sentence and adds the fixations as dots on the words. Each dot representing a fixation is
    connected to the next fixation by an arc. The words IDS below the sentence denotes the fixation sequence.
    """
    scanpath_dict = json.load(open(file_path))

    example_scanpaths = []
    predicted_word_idxs = []
    true_word_idxs = []

    # filter sentence that are shorter or equal to max_words
    for sentence, fixation_indices, true_fixation_indices in zip(
            scanpath_dict['original_sn'],
            scanpath_dict['predicted_sp_ids'],
            scanpath_dict['original_sp_ids']
    ):

        sent = sentence.split()

        if len(sent) <= max_words and len(example_scanpaths) < num_sample_sp:
            example_scanpaths.append(sentence)
            predicted_word_idxs.append(fixation_indices)
            true_word_idxs.append(true_fixation_indices)

    # plot the sentence on a white background using the pillow library
    for idx, (sent, predicted_ids, true_ids) in tqdm(
            enumerate(zip(example_scanpaths, predicted_word_idxs, true_word_idxs)),
            desc=f'Plotting scanpaths',
            total=len(example_scanpaths)
    ):
        plot_scanpath_image(sent, predicted_ids, idx, img_size, letter_height, 'predicted', rank, batch_id, output_dir)
        plot_scanpath_image(sent, true_ids, idx, img_size, letter_height, 'true', rank, batch_id, output_dir)


def plot_scanpath_image(
        sentence: str,
        scanpath: list,
        instance_id: int,
        img_size: tuple = (1400, 600),
        text_height: int = 44,
        name: str = None,
        rank: int = None,
        batch_id: int = None,
        output_dir: str = None,
):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    font_size_small = int(text_height * SCALING_FACTOR - (text_height // 3.5) * SCALING_FACTOR)
    font_size_legend = int(text_height * SCALING_FACTOR - (text_height // 5) * SCALING_FACTOR)

    try:
        monospace = ImageFont.truetype('Courier New.ttf', text_height * SCALING_FACTOR)
        # monospace = ImageFont.truetype('visualizations/Raleway-Regular.ttf', text_height * SCALING_FACTOR)
        monospace_small = ImageFont.truetype('Courier New.ttf', font_size_small)
        legend_font = ImageFont.truetype('Times New Roman.ttf', font_size_legend)
        # monospace_small = ImageFont.truetype('visualizations/Raleway-Regular.ttf', font_size_small)
    except OSError:
        try:
            monospace = ImageFont.truetype('cour.ttf', text_height * SCALING_FACTOR)
            monospace_small = ImageFont.truetype('cour.ttf', font_size_small)
            legend_font = ImageFont.truetype('cour.ttf', font_size_legend)
        except OSError:
            raise OSError(
                'Could not find the right font. '
                'Please find out how "courier new" is called on your system'
            )

    img_width, img_height, len_padding = _get_image_size(sentence, img_size, monospace)

    img_size = (img_width, img_height)

    image = Image.new('RGBA', img_size, color=(255, 255, 255, 255))

    color_mapping = {
        'forward_saccade': (216, 27, 96),
        'regression': (30, 136, 229),
        'fixation': (255, 193, 7, 140)
    }

    legend_image = Image.new('RGBA', img_size, color=(255, 255, 255, 0))
    legend_draw_object = ImageDraw.Draw(legend_image)
    _draw_legend(color_mapping, legend_draw_object, legend_font,
                 img_size,
                 x_top_left=(len_padding // 15),
                 # y_top_left=img_size[1] - (font_size_small*(0.2+len(color_mapping))),
                 y_top_left=font_size_small * 7.5,
                 orientation='horizontal',
                 )

    image = Image.alpha_composite(image, legend_image)

    draw_object = ImageDraw.Draw(image)

    word_centers_px = {}

    x_px = len_padding
    y_px = (img_size[1] // 2) + (text_height * SCALING_FACTOR)

    if name == 'predicted':
        draw_object.text(((len_padding // 15), y_px), 'SᴄᴀɴDL: ', fill='black', font=monospace)
    else:
        draw_object.text(((len_padding // 15), y_px - text_height * 0.7 * SCALING_FACTOR), 'Human\nScanpath: ',
                         fill='black', font=monospace)
    draw_object.text(
        ((len_padding // 15), y_px + text_height * 1.5 * SCALING_FACTOR), 'Fixation index: ', fill='black',
        font=monospace_small
    )

    for word_idx, word in enumerate(sentence.split()):
        # get fixation indices for the current word
        fix_index_word = [i + 1 for i, word in enumerate(scanpath) if word == word_idx]
        fix_idx_str = '\n'.join([str(n) for n in fix_index_word]).strip()

        word = word + " "
        word_width = draw_object.textlength(word, font=monospace)

        word_center = (x_px + word_width // 2, y_px + (text_height * SCALING_FACTOR) // 2)
        word_centers_px[word_idx] = word_center

        # draw fix indices below words
        if fix_idx_str:
            fix_width = draw_object.textlength(fix_idx_str.strip().split()[-1], font=monospace_small)
            draw_object.text(
                (word_center[0] - (fix_width // 2), y_px + text_height * 1.55 * SCALING_FACTOR),
                fix_idx_str,
                fill='black',
                font=monospace_small
            )

        x_px += word_width

    alpha_image = Image.new('RGBA', img_size, color=(255, 255, 255, 0))
    angle = 80
    alpha_draw_object = ImageDraw.Draw(alpha_image)
    fixation_angles = {}

    for fix_idx, fixation in enumerate(scanpath[:-1]):
        source = (word_centers_px[fixation][0],
                  word_centers_px[fixation][1])
        target = (word_centers_px[scanpath[fix_idx + 1]][0],
                  word_centers_px[scanpath[fix_idx + 1]][1])

        pair = f'{fixation}{scanpath[fix_idx + 1]}'

        # if there are multiple saccades between the same two words, increase the angle
        if pair not in fixation_angles and not pair[::-1] in fixation_angles:
            fixation_angles[pair] = angle
            fixation_angles[pair[::-1]] = angle
        elif source != target:
            fixation_angles[pair] += 25
            fixation_angles[pair[::-1]] += 25

        _draw_fixations_arcs(
            draw_object,
            alpha_draw_object,
            *source,
            *target,
            angle=fixation_angles[pair],
            color_maps=color_mapping,
            text_size=text_height,
        )

    x_px = len_padding

    draw_object.text((x_px, y_px), sentence, fill='black', font=monospace)

    image = Image.alpha_composite(image, alpha_image)

    im = image.resize((img_size[0] // SCALING_FACTOR, img_size[1] // SCALING_FACTOR), resample=Image.LANCZOS)

    filename = f'sp_plot_rank{rank}_batch{batch_id}_instance{instance_id}_{name}.png'
    file_path = os.path.join(output_dir, filename)

    # SvgImagePlugin.draw_svg_file(filename, draw_object, img_size, '#00000000')

    im.save(
        file_path,
        dpi=(1200, 1200),
    )


def _get_image_size(sentence: str, img_size: tuple, font: ImageFont) -> tuple:
    img_size = (img_size[0] * SCALING_FACTOR, img_size[1] * SCALING_FACTOR)

    image = Image.new('RGBA', img_size, color=(255, 255, 255, 255))
    draw_object = ImageDraw.Draw(image)

    sent_length = int(draw_object.textlength(sentence, font=font))
    len_padding = int(draw_object.textlength('Sentence:     ', font=font))
    img_width = sent_length + len_padding

    return img_width, img_size[1], len_padding


def _draw_fixations_arcs(
        draw: ImageDraw.Draw,
        alpha_draw: ImageDraw.Draw,
        x1: int, y1: int, x2: int, y2: int,
        angle: int, color_maps: dict,
        text_size: int,
):
    pos_x1 = x1
    pos_x2 = x2
    radius_2 = 0

    outline_color = color_maps['forward_saccade']
    outline_width = (text_size // 11) * SCALING_FACTOR

    if x2 < x1:
        x2, x1 = x1, x2
        y2, y1 = y1, y2
        outline_color = color_maps['regression']

    if x2 == x1:
        x2 += (text_size * 0.8) * SCALING_FACTOR
        x1 -= (text_size * 0.8) * SCALING_FACTOR
        angle = 180
        radius_2 = -(text_size // 3) * SCALING_FACTOR

    half_angle = angle // 2
    half_dist = (x2 - x1) // 2

    # Set the center and radius of the circle (sine rule + some pythagoras)
    radius = (math.sin(math.radians(90)) / math.sin(math.radians(half_angle))) * half_dist
    radius_2 += radius
    b = math.sqrt(radius ** 2 - half_dist ** 2)
    center = (x2 - half_dist, y2 + b)

    # get the complementary angles for the start and end angles
    start_angle = math.degrees(math.atan2(y2 - center[1], x2 - center[0])) % 360
    end_angle = math.degrees(math.atan2(y1 - center[1], x1 - center[0])) % 360

    # Draw the upper portion of the circle above the points
    draw.arc(
        (center[0] - radius_2, center[1] - radius, center[0] + radius_2, center[1] + radius),
        start=end_angle, end=start_angle, fill=outline_color, width=outline_width
    )

    # Draw fixations using filled circles
    point_radius = (text_size // 2) * SCALING_FACTOR

    alpha_draw.ellipse(
        (pos_x1 - point_radius, y1 - point_radius, pos_x1 + point_radius, y1 + point_radius),
        fill=color_maps['fixation']
    )
    alpha_draw.ellipse(
        (pos_x2 - point_radius, y2 - point_radius, pos_x2 + point_radius, y2 + point_radius),
        fill=color_maps['fixation']
    )


def _draw_legend(color_text_mapping: dict, draw_obj: ImageDraw, font: ImageFont, image_size: tuple,
                 x_top_left: float, y_top_left: float, orientation: str = 'vertical',
                 ):
    radius = font.size * 0.2
    x_text = x_top_left + 2 * radius + font.size * 0.45
    line_width = font.size // 6
    space = draw_obj.textlength('  ', font=font)

    if orientation == 'horizontal':
        for text, color in color_text_mapping.items():

            entry_len = 0
            first_y = y_top_left + 18 * SCALING_FACTOR
            second_y = y_top_left + 18 * SCALING_FACTOR + 2 * radius

            if text == 'forward_saccade':
                text = 'Progressive saccade'
                entry_len = draw_obj.textlength(text, font=font) + (8 * radius)

                draw_obj.line((x_top_left,
                               first_y + line_width,
                               x_top_left + 2 * radius,
                               first_y + line_width), fill=color, width=line_width)

            elif text == 'regression':
                text = 'Regressive saccade'
                entry_len = draw_obj.textlength(text, font=font) + (8 * radius)

                draw_obj.line((x_top_left,
                               first_y + line_width,
                               x_top_left + 2 * radius,
                               first_y + line_width), fill=color, width=line_width)

            elif text == 'fixation':
                text = 'Fixation'
                entry_len = draw_obj.textlength(text, font=font) + (8 * radius)

                draw_obj.ellipse((x_top_left,
                                  first_y,
                                  x_top_left + 2 * radius,
                                  second_y), fill=color)

            draw_obj.text((x_text, y_top_left), text, fill='black', font=font)
            x_top_left += entry_len
            x_text += entry_len

    if orientation == 'vertical':

        for text, color in color_text_mapping.items():

            first_y = y_top_left + 18 * SCALING_FACTOR
            second_y = y_top_left + 18 * SCALING_FACTOR + 2 * radius

            if text == 'forward_saccade':
                text = 'Progressive saccade'
                draw_obj.line((x_top_left,
                               first_y + line_width,
                               x_top_left + 2 * radius,
                               first_y + line_width), fill=color, width=line_width)

            elif text == 'regression':
                text = 'Regressive saccade'
                draw_obj.line((x_top_left,
                               first_y + line_width,
                               x_top_left + 2 * radius,
                               first_y + line_width), fill=color, width=line_width)

            elif text == 'fixation':
                text = 'Fixation'
                draw_obj.ellipse((x_top_left,
                                  first_y,
                                  x_top_left + 2 * radius,
                                  second_y), fill=color)

            draw_obj.text((x_text, y_top_left), text, fill='black', font=font)
            y_top_left += font.size


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--file-path',
        default='seed70_step0_clamp-first-yes_all_remove-PAD.json',
        help='json file containing predicted words, fixation word ids and the original sentence'
    )

    parser.add_argument(
        '-m', '--max-words',
        default=5,
    )

    parser.add_argument(
        '-n', '--num-sample-sp',
        default=5,
    )

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    sample_and_plot_scanpaths(**args)
