from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import torch
from IPython.core.display import display, HTML
from matplotlib import colors
from tokenizers import Tokenizer
from torch import Tensor

from model import ClassificationAttentionModel
from utils import label_string


class Visualizer:
    def __init__(self, model: ClassificationAttentionModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # based on https://github.com/jiesutd/Text-Attention-Heatmap-Visualization/blob/master/text_attention.py
    def to_latex(self, text: str, file_path: Union[Path, str], color: str = 'red'):
        _, attention_weights = self.__forward(text) * 100
        words = self.__clean_text(text)

        with open(file_path, 'w') as f:
            f.write(
                r'''
                    \documentclass[varwidth]{standalone}
                    \special{papersize=210mm,297mm}
                    \usepackage{color}
                    \usepackage{tcolorbox}
                    \usepackage{CJK}
                    \usepackage{adjustbox}
                    \tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
                    \begin{document}
                    \begin{CJK*}{UTF8}{gbsn}
                '''.strip() + '\n'
            )

            string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{''' + '\n'

            for i in range(len(words)):
                string += '\\colorbox{%s!%s}{' % (color, attention_weights[i]) + '\\strut ' + words[i] + '} '

            string += '\n}}}\n'
            f.write(string.strip())
            f.write(
                r'''
                    \end{CJK*}
                    \end{document}
                ''')

    def to_html(self, text: str, true_label: Union[None, int] = None, color: str = 'red') -> Tuple[int, str]:
        prediction, attention_weights = self.__forward(text)
        words = text.split(' ')
        rgb = ','.join(map(lambda x: str(x * 255), colors.to_rgb(color)))

        string = f'<span>Prediction: {label_string(prediction)}</span></br>'

        if true_label is not None:
            string += f'<span>True label: {label_string(true_label)}</span></br>'

        for i in range(len(words)):
            string += f'<span style="background-color:rgba({rgb}, {attention_weights[i]});">{words[i]}</span> '

        return prediction, string

    def show_html(self, text: str, true_label: Union[None, int] = None, color: str = 'red'):
        _, html = self.to_html(text, true_label, color)
        display(HTML(html))

    @torch.no_grad()
    def __forward(self, text: str) -> Tuple[int, np.ndarray]:
        input_ids = Tensor(self.tokenizer.encode(text).ids).unsqueeze(0)
        prediction, attention_weights = self.model((input_ids, Tensor([input_ids.shape[-1]])))
        return prediction.cpu().argmax().item(), attention_weights.cpu().squeeze().numpy()

    @staticmethod
    def __clean_text(text: str) -> List[str]:
        words = []
        for word in text.split(' '):
            for latex_sensitive in ['\\', '%', '&', '^', '#', '_', '{', '}']:
                if latex_sensitive in word:
                    word = word.replace(latex_sensitive, '\\' + latex_sensitive)
            words.append(word)

        return words