#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mono Tool - libcstを使用したPythonファイル編集ツール

シンプルなPythonファイル編集ツール。
libcstを使用して関数やメソッドを編集します。
"""

import libcst as cst
import black
from langchain_core.tools import tool
from typing import Any, List
import os
import shutil
import inspect


def simple_function():
    """This is a simple function that returns a greeting.

    Returns:
        str: A friendly greeting message.
    """
    return "Hello, World!"


class Tool:
    """ツールの基底クラス"""

    def __init__(self, settings=None, app_data=None):
        """
        ツールの初期化

        Args:
            settings: ツールの設定情報
            app_data: AppDataへの参照
        """
        self.settings = settings or {}
        self.app_data = app_data
        self.functions = []

        # ファイルパスの設定
        self.file_path = settings.get("file_path", "tools/mono_tool.py")

    def get_file_path(self):
        """ファイルパスを取得"""
        return self.file_path


def normalize_with_black(content: str) -> str:
    """
    blackを使用してPythonコードを正規化する

    Args:
        content: 正規化対象のPythonコード

    Returns:
        正規化されたPythonコード
    """
    return black.format_str(content, mode=black.FileMode())


def create_backup_file(file_path: str) -> str:
    """
    ファイルのバックアップを作成する

    Args:
        file_path: バックアップを作成するファイルのパス

    Returns:
        バックアップファイルのパス
    """
    # パスとファイル名を分離
    dir_path, filename = os.path.split(file_path)

    # ファイル名と拡張子を分離
    name, ext = os.path.splitext(filename)

    # バックアップファイル名を作成
    backup_filename = f"{name}_backup{ext}"
    backup_path = os.path.join(dir_path, backup_filename)

    # ファイルをコピー
    shutil.copy2(file_path, backup_path)

    return backup_path


class FunctionEditor(cst.CSTTransformer):
    """関数を編集するTransformer"""

    def __init__(self, target_name: str, new_code: str):
        self.target_name = target_name
        self.new_code = new_code
        self.success = False

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if updated_node.name.value == self.target_name:
            # 新しいコードをパースして置き換え
            new_func = cst.parse_statement(self.new_code)
            self.success = True
            return new_func
        return updated_node


class MethodEditor(cst.CSTTransformer):
    """クラスメソッドを編集するTransformer"""

    def __init__(self, class_name: str, method_name: str, new_code: str):
        self.class_name = class_name
        self.method_name = method_name
        self.new_code = new_code
        self.in_target_class = False
        self.success = False

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.in_target_class = node.name.value == self.class_name

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self.in_target_class = False
        return updated_node

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if self.in_target_class and updated_node.name.value == self.method_name:
            new_method = cst.parse_statement(self.new_code)
            self.success = True
            return new_method
        return updated_node


class ClassEditor(cst.CSTTransformer):
    """クラス定義を編集するTransformer"""

    def __init__(self, target_name: str, new_code: str):
        self.target_name = target_name
        self.new_code = new_code
        self.success = False

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if updated_node.name.value == self.target_name:
            # 新しいコードをパースして置き換え
            new_class = cst.parse_statement(self.new_code)
            self.success = True
            return new_class
        return updated_node


def edit_python_element(
    file_path: str, element_name: str, new_code: str, element_type: str = "function"
):
    """
    Pythonファイルの関数、メソッド、クラスを編集する汎用関数

    Args:
        file_path: 編集対象ファイルのパス
        element_name: 編集する要素名 (関数名、クラス名、"クラス名.メソッド名")
        new_code: 新しいコード
        element_type: "function", "method", "class"

    Returns:
        bool: 編集が成功したかどうか
    """
    # ファイル読み込み
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    # 編集前の正規化（必須）
    normalized_source = normalize_with_black(source_code)

    # CSTに変換
    tree = cst.parse_module(normalized_source)

    # 編集実行
    if element_type == "function":
        transformer = FunctionEditor(element_name, new_code)
    elif element_type == "method":
        if "." not in element_name:
            raise ValueError(
                "メソッド編集の場合は 'クラス名.メソッド名' 形式で指定してください"
            )
        class_name, method_name = element_name.split(".", 1)
        transformer = MethodEditor(class_name, method_name, new_code)
    elif element_type == "class":
        transformer = ClassEditor(element_name, new_code)
    else:
        raise ValueError(f"未対応の要素タイプ: {element_type}")

    modified_tree = tree.visit(transformer)

    if not transformer.success:
        raise ValueError(f"要素 '{element_name}' が見つかりませんでした")

    # 編集後の正規化（必須）
    result_code = normalize_with_black(modified_tree.code)

    # ファイル保存
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result_code)

    return True


def edit_header_section(file_path: str, new_header: str):
    """
    特殊コメント「# === MONO_TOOL_HEADER_END ===」までの内容を置き換える

    Args:
        file_path: 編集対象ファイルのパス
        new_header: 新しいヘッダー内容（import文、docstring、グローバル変数など）

    Returns:
        bool: 編集が成功したかどうか
    """
    HEADER_END_MARKER = "# === MONO_TOOL_HEADER_END ==="

    # ファイル読み込み
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    # 特殊コメントを探す
    lines = source_code.split("\n")
    marker_index = -1

    for i, line in enumerate(lines):
        if HEADER_END_MARKER in line:
            marker_index = i
            break

    if marker_index == -1:
        raise ValueError(f"特殊コメント '{HEADER_END_MARKER}' が見つかりません")

    # ヘッダー部分を置き換え
    new_content = new_header.rstrip("\n") + "\n\n" + HEADER_END_MARKER + "\n"
    remaining_content = "\n".join(lines[marker_index + 1 :])

    full_content = new_content + remaining_content

    # black正規化（必須）
    normalized_content = normalize_with_black(full_content)

    # ファイル保存
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(normalized_content)

    return True


def edit_special_block(file_path: str, new_block: str):
    """
    if __name__ == "__main__": ブロックを置き換える

    Args:
        file_path: 編集対象ファイルのパス
        new_block: 新しい特殊ブロック内容

    Returns:
        bool: 編集が成功したかどうか
    """
    # ファイル読み込み
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    # if __name__ == "__main__": を探す
    lines = source_code.split("\n")
    main_block_start = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("if __name__") and "__main__" in stripped:
            main_block_start = i
            break

    if main_block_start == -1:
        # if __name__ == "__main__": ブロックが存在しない場合は末尾に追加
        new_content = source_code.rstrip("\n") + "\n\n\n" + new_block
    else:
        # 既存のif __name__ == "__main__": ブロックを置き換え
        before_main = "\n".join(lines[:main_block_start])
        new_content = before_main.rstrip("\n") + "\n\n\n" + new_block

    # black正規化（必須）
    normalized_content = normalize_with_black(new_content)

    # ファイル保存
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(normalized_content)

    return True


@tool
def edit_mono(
    element_name: str,
    element_type: str,
    new_content: str,
    file_path: str = None,
    create_backup: bool = False,
) -> str:
    """
    Pythonファイルの要素を編集します。
    編集前後でblack正規化を実行してフォーマットを統一します。
    create_backup=Trueの場合、編集前に自動的にバックアップファイルを作成します。

    Args:
        element_name: 編集する要素名
          - 関数: 関数名
          - メソッド: "クラス名.メソッド名"
          - クラス: クラス名
          - ヘッダー: "header" (固定)
          - 特殊ブロック: "main_block" (固定)
        element_type: 要素タイプ
          - "function": 関数
          - "method": クラスメソッド
          - "class": クラス定義
          - "header": ヘッダー部分（import文、docstring、グローバル変数）
          - "special_block": 特殊構文ブロック（if __name__ == "__main__":）
        new_content: 新しいコード内容
        file_path: 編集対象ファイルのパス（指定がない場合はツール設定から取得）
        create_backup: バックアップファイルを作成するかどうか（デフォルト: False）

    Returns:
        str: 編集結果のメッセージ
    """
    try:
        # ファイルパスが指定されていない場合は、デフォルトパスを使用
        if file_path is None:
            # デフォルトのファイルパスを使用
            file_path = "tools/mono_tool/tool.py"

        # ファイルが存在するか確認
        if not os.path.exists(file_path):
            return f"ERROR: ファイルが存在しません: {file_path}"

        # バックアップ作成（編集処理の前に実行）
        backup_path = ""
        backup_msg = ""
        if create_backup:
            backup_path = create_backup_file(file_path)
            backup_msg = f"\nバックアップファイル: {backup_path}"

        # element_typeの正規化と処理分岐
        if element_type in ["class_method", "method"]:
            element_type = "method"
            edit_python_element(
                file_path=file_path,
                element_name=element_name,
                new_code=new_content,
                element_type=element_type,
            )
            return (
                f"OK: {element_name} の編集が完了しました{backup_msg}\nblack正規化適用"
            )

        elif element_type == "function":
            edit_python_element(
                file_path=file_path,
                element_name=element_name,
                new_code=new_content,
                element_type=element_type,
            )
            return (
                f"OK: {element_name} の編集が完了しました{backup_msg}\nblack正規化適用"
            )

        elif element_type == "class":
            edit_python_element(
                file_path=file_path,
                element_name=element_name,
                new_code=new_content,
                element_type=element_type,
            )
            return (
                f"OK: {element_name} の編集が完了しました{backup_msg}\nblack正規化適用"
            )

        elif element_type == "header":
            edit_header_section(file_path=file_path, new_header=new_content)
            return f"OK: ヘッダー部分の編集が完了しました{backup_msg}\nblack正規化適用"

        elif element_type == "special_block":
            edit_special_block(file_path=file_path, new_block=new_content)
            return f"OK: 特殊ブロックの編集が完了しました{backup_msg}\nblack正規化適用"

        else:
            return f"ERROR: 未対応の要素タイプです: {element_type}\n対応タイプ: function, method, class, header, special_block"

    except Exception as e:
        return f"ERROR: 編集に失敗しました - {str(e)}"
