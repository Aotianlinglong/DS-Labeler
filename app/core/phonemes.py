# -*- coding: utf-8 -*-
"""
音素工具：语言标签解析 + 元音/辅音分类（按语言前缀分流）。

音素名格式（与 data_out 真实数据一致）：
  - 全局特殊音素：AP / SP / EP / GS / sil / br / pau（无前缀，不参与元辅判断）
  - 带语言前缀：`en/d`、`yue/n`（前缀 ∈ zh/ja/en/yue，按对应语言元辅表分类）
  - 无前缀：按工程主语言元辅表分类
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

# 内置四个语言
LANGS = ("zh", "ja", "en", "yue")

# 默认全局特殊音素（可自定义增删）
DEFAULT_GLOBAL_PHONEMES = ["AP", "SP", "EP", "GS", "sil", "br", "pau"]

# ---- 三种音素系统（DiffSinger 官方划分方式） ---------------------------
SYSTEM_BINARY = "binary"     # 二分式字典（汉语普通话 / 日语）
SYSTEM_MONO = "mono"         # 单音节音素系统（粤语 / 韩语）
SYSTEM_POLY = "poly"         # 多音节音素系统（英语 / 俄语）
SYSTEMS = (SYSTEM_BINARY, SYSTEM_MONO, SYSTEM_POLY)
SYSTEM_NAMES = {
    SYSTEM_BINARY: "二分式",
    SYSTEM_MONO: "单音节",
    SYSTEM_POLY: "多音节",
}
# 各语言默认系统：zh/ja 二分式，yue 单音节，en 多音节
DEFAULT_SYSTEMS = {
    "zh": SYSTEM_BINARY, "ja": SYSTEM_BINARY,
    "yue": SYSTEM_MONO, "en": SYSTEM_POLY,
}

# 元辅表 / 流音表**不再是代码常量**：四个语言（zh / ja / en / yue）都从
# `assets/dictionaries/<lang>/{vowels,consonants,liquids}.txt` 读（见 core.paths.lang_dict）。
# 这样用户可以直接编辑 / 增删音素，不必改代码（README「音素词典」一节）。

# 符号类型表（如 DiffSinger 训练配置的 phones.txt）type 字段 → 分类
TYPE_TO_CATEGORY = {
    "vowel": "vowel", "diphthong": "vowel",
    "liquid": "liquid",
    "fricative": "consonant", "consonant": "consonant", "plosive": "consonant",
    "affricate": "consonant", "nasal": "consonant", "approximant": "consonant",
    "silence": "global", "global": "global",
}


class SymbolTypeTable:
    """符号类型表：`symbol -> 分类` 的显式映射（优先级最高的音素类型来源）。

    支持 DiffSinger 训练配置里的 phones.txt（YAML：顶层 `symbols:` 列表，
    每项 `{symbol, type}`）。
    """

    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping: dict[str, str] = dict(mapping or {})

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "SymbolTypeTable | None":
        """读取类型表；文件缺失 / 解析失败返回 None（调用方优雅回退）。"""
        import yaml

        p = pathlib.Path(path)
        if not str(path).strip() or not p.exists():
            return None
        try:
            data = yaml.safe_load(p.read_text(encoding="utf8")) or {}
        except Exception:
            return None
        mapping: dict[str, str] = {}
        for item in (data.get("symbols") or []):
            if not isinstance(item, dict):
                continue
            sym, typ = item.get("symbol"), item.get("type")
            if not sym:
                continue
            mapping[str(sym)] = TYPE_TO_CATEGORY.get(str(typ), "consonant")
        return cls(mapping) if mapping else None

    def category(self, ph: str) -> str | None:
        return self.mapping.get(ph)


@dataclass
class LanguageTable:
    """一种语言的元音/辅音/流音表"""
    name: str
    vowels: set = field(default_factory=set)
    consonants: set = field(default_factory=set)
    liquids: set = field(default_factory=set)

    def load_from_files(self, vowels_path=None, consonants_path=None,
                        liquids_path=None) -> None:
        """从 txt 载入元音 / 辅音 / 流音（空格或换行分隔）。

        缺文件 / 空路径一律跳过（该表留空），故调用方可以直接把 `paths.lang_dict()`
        的返回值丢进来，不必先判存在性。
        """
        for path, target in ((vowels_path, self.vowels),
                             (consonants_path, self.consonants),
                             (liquids_path, self.liquids)):
            if not path:
                continue
            p = pathlib.Path(path)
            if not p.exists():
                continue
            with open(p, "r", encoding="utf8") as f:
                target.update(f.read().split())

    @property
    def overlap(self) -> set:
        return self.vowels & self.consonants


class PhonemeClassifier:
    """
    元辅判断环节（设置内自动化，表面不暴露）。
    音素类型来源优先级：
      1. 符号类型表（如 phones.txt：显式 symbol → 类型）
      2. 该语言的元音 / 辅音 / 流音表（自定义 txt 或内置表）
    规则：
      - 全局特殊音素 → global（不做元辅判断）
      - 带 `<lang>/` 前缀 → 按该语言表
      - 无前缀 → 按主语言表
    """

    def __init__(
        self,
        main_lang: str = "zh",
        tables: dict[str, LanguageTable] | None = None,
        global_phonemes: list[str] | None = None,
        systems: dict[str, str] | None = None,
        symbol_types: SymbolTypeTable | None = None,
    ):
        assert main_lang in LANGS, f"主语言必须是 {LANGS} 之一，收到 {main_lang!r}"
        self.main_lang = main_lang
        if tables is None:
            # 默认表：直接读四个语言的内置词典（不持有工程时也能正确分类）
            from core.paths import lang_dict
            tables = {}
            for lang in LANGS:
                t = LanguageTable(lang)
                t.load_from_files(*lang_dict(lang))
                tables[lang] = t
        self.tables: dict[str, LanguageTable] = tables
        self.global_phonemes = set(global_phonemes or DEFAULT_GLOBAL_PHONEMES)
        self.systems: dict[str, str] = dict(systems or {})
        self.symbol_types: SymbolTypeTable | None = symbol_types

    # ---- 音素系统 -----------------------------------------------------
    def system_of(self, lang: str | None) -> str:
        """某语言的音素系统（未配置 → 内置默认，再退化为二分式）。"""
        if lang not in LANGS:
            lang = self.main_lang
        return self.systems.get(lang) or DEFAULT_SYSTEMS.get(lang, SYSTEM_BINARY)

    # ---- 音素名解析 ---------------------------------------------------
    @staticmethod
    def split_phoneme(ph: str) -> tuple[str | None, str]:
        """返回 (语言标签或 None, 音素本体)。全局音素返回 (None, ph)。"""
        if "/" in ph:
            lang, _, name = ph.partition("/")
            return (lang if lang in LANGS else None), name
        return None, ph

    # ---- 分类 ---------------------------------------------------------
    def classify(self, ph: str) -> str:
        """
        返回属性：
          'global'    —— 全局特殊音素
          'vowel'     —— 元音
          'liquid'    —— 流音（多音节系统需要）
          'consonant' —— 辅音
          'unknown'   —— 在对应语言表里找不到（导入时高亮提示）
          'bad_lang'  —— 语言标签不存在（导入时高亮提示）
        """
        if ph in self.global_phonemes:
            return "global"
        if self.symbol_types is not None:      # ① 符号类型表优先
            cat = self.symbol_types.category(ph)
            if cat is not None:
                return cat
        lang, name = self.split_phoneme(ph)
        if lang is None:
            lang = self.main_lang
        table = self.tables.get(lang)
        if table is None:
            return "bad_lang"
        if name in table.vowels:
            return "vowel"
        if name in table.liquids:
            return "liquid"
        if name in table.consonants:
            return "consonant"
        return "unknown"

    def classify_seq(self, ph_seq: list[str]) -> list[str]:
        return [self.classify(p) for p in ph_seq]

    # ---- 未知音素收集（导入校验用） ------------------------------------
    def unknown_phonemes(self, ph_seq: list[str]) -> list[str]:
        return [p for p in ph_seq if self.classify(p) == "unknown"]

    def bad_lang_phonemes(self, ph_seq: list[str]) -> list[str]:
        return [p for p in ph_seq if self.classify(p) == "bad_lang"]

    # ---- ph_num（按音素系统计算） ---------------------------------------
    def _is_consonant(self, ph: str) -> bool:
        """二分式 / 单音节里「辅音」含流音（流音仅在多音节系统里单列）。"""
        return self.classify(ph) in ("consonant", "liquid")

    def _binary_flags(self, ph_seq: list[str]) -> list[bool]:
        """词 = [一个音素 + 后续辅音串]；返回每个词起点标记（与 add_ph_num.py 一致）。"""
        flags: list[bool] = []
        i, n = 0, len(ph_seq)
        while i < n:
            flags.append(True)
            j = i + 1
            while j < n and self._is_consonant(ph_seq[j]):
                j += 1
            flags.extend([False] * (j - i - 1))
            i = j
        return flags

    def _poly_type(self, ph: str) -> str:
        c = self.classify(ph)
        if c in ("vowel", "global"):
            return "V"
        if c == "liquid":
            return "L"
        if c == "consonant":
            return "C"
        return "X"

    def _poly_flags(self, ph_seq: list[str]) -> list[bool]:
        """多音节：三条 trie 规则 `*V` / `*C *L *V` / `*L *L *V`（与 add_ph_num_advanced.py 一致）。

        三条规则都用通配符词项，故「最佳路径」= 最长匹配：优先 3 长度的 C/L-L-V，
        否则单 V。命中的音素位（syllable onset）标 True。
        """
        types = [self._poly_type(p) for p in ph_seq]
        flags: list[bool] = []
        i, n = 0, len(types)
        while i < n:
            t = types[i]
            if (i + 2 < n and types[i + 1] == "L" and types[i + 2] == "V"
                    and t in ("C", "L")):
                flags.extend([False, True, False])
                i += 3
            elif t == "V":
                flags.append(True)
                i += 1
            else:
                flags.append(False)
                i += 1
        return flags

    @staticmethod
    def _accumulate(flags: list[bool]) -> list[int]:
        """起点标记 → ph_num（每个词所辖音素数）。"""
        acc = 0
        ph_num: list[int] = []
        for f in flags:
            if f:
                if acc > 0:
                    ph_num.append(acc)
                acc = 1
            else:
                acc += 1
        if acc > 0:
            ph_num.append(acc)
        return ph_num

    def _seq_lang(self, ph_seq: list[str]) -> str:
        for p in ph_seq:
            lg, _ = self.split_phoneme(p)
            if lg in LANGS:
                return lg
        return self.main_lang

    def _words_all_poly(self, ph_seq: list[str], word_sizes: list[int]) -> bool:
        """整句是否**每个词**都属于多音节系统（决定要不要按词走 trie）。

        官方两个脚本都只按**整句口径**选一种：`add_ph_num.py` 全局扫描；
        `add_ph_num_advanced.py` 逐词 trie（但 accumulate 仍是全句一次）。
        混合句（zh 数据里夹 `en/` 音素等）按词分块会**拆断**「元音 + 后接辅音串」的
        全局扫描：每个汉字 2 音素被切成 2 组（`_binary_flags` 无条件把词首标成组头），
        ph_num 组数涨到音素量级 —— 那既非官方两脚本任一口径，也与本类的注释相悖。
        故：只有**全部**词都是多音节时才走按词 trie，否则整句退回全局二分扫描。
        """
        i = 0
        for k in word_sizes:
            k = int(k)
            if k <= 0:
                continue
            word = ph_seq[i:i + k]
            i += k
            if word and self.system_of(self._seq_lang(word)) != SYSTEM_POLY:
                return False
        tail = ph_seq[i:]
        return not tail or self.system_of(self._seq_lang(tail)) == SYSTEM_POLY

    def compute_ph_num(self, ph_seq: list[str],
                       word_sizes: list[int] | None = None) -> list[int]:
        """DiffSinger ph_num = 每个词所辖音素数（Σ 恒等于音素总数）。

        - **多音节**（en/ru）：需要 words 层分组，按官方三条 trie 规则
          （add_ph_num_advanced.py）；word_sizes 为每词音素数（由 words 层反推）。
        - **二分式 / 单音节**（zh/ja/yue）：走「元音 + 后接辅音串」全局扫描
          （与 add_ph_num.py 一致），**忽略 word_sizes** —— 源 words 层是「音节」划分，
          与算法划分不同，若按词分块会算错，必须与原流水线逐字符一致。
        - **混合句**：只要有一个词不是多音节，整句都走全局扫描（见 `_words_all_poly`）。
        """
        if word_sizes and self._words_all_poly(ph_seq, word_sizes):
            flags: list[bool] = []
            i = 0
            for k in word_sizes:
                k = int(k)
                if k <= 0:
                    continue
                word = ph_seq[i:i + k]
                i += k
                if not word:
                    continue
                if self.system_of(self._seq_lang(word)) == SYSTEM_POLY:
                    flags.extend(self._poly_flags(word))
                else:
                    flags.extend(self._binary_flags(word))
            if i < len(ph_seq):      # 尾部未被词覆盖的音素按二分式处理
                flags.extend(self._binary_flags(ph_seq[i:]))
            if flags:
                return self._accumulate(flags)
        return self._accumulate(self._binary_flags(ph_seq))
