#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Mono Tool - mono_tool.pyとmono_tool_backup.pyの比較テスト

このツールは、mono_tool.pyとmono_tool_backup.pyが一致することを確認します。

使用方法:
    Craft AI内で #test_mono  # AI用ツール実行
"""

import unittest
import subprocess
import sys
import os
import json
from pathlib import Path
from langchain_core.tools import tool
import inspect


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
        
        # mono_toolのパスを取得
        self.mono_tool_path = None
        if app_data and hasattr(app_data, 'tools') and 'mono_tool' in app_data.tools:
            mono_tool = app_data.tools['mono_tool']
            if hasattr(mono_tool, 'file_path'):
                self.mono_tool_path = mono_tool.file_path
        
        # パスが取得できなかった場合はデフォルト値を使用
        if not self.mono_tool_path:
            self.mono_tool_path = "tools/mono_tool.py"
    
    def get_mono_tool_path(self):
        """mono_toolのパスを取得"""
        return self.mono_tool_path


class TestMonoToolComparison(unittest.TestCase):
    """mono_tool.py vs mono_tool_backup.py ファイル比較テスト"""

    def setUp(self):
        """テスト前の準備"""
        # 現在のファイルのディレクトリを取得
        current_dir = Path(__file__).parent.parent.parent  # craft-aiディレクトリ
        
        # mono_toolのパスを設定
        self.mono_tool_py = current_dir / "tools" / "mono_tool" / "tool.py"
        
        # バックアップファイルのパスを設定
        self.mono_tool_backup_py = current_dir / "tools" / "mono_tool" / "tool_backup.py"
        


    def test_backup_file_exists(self):
        """バックアップファイルが存在することを確認"""
        self.assertTrue(self.mono_tool_backup_py.exists(), 
                       f"バックアップファイルが存在しません: {self.mono_tool_backup_py}")

    def test_mono_tool_exists(self):
        """mono_tool.pyが存在することを確認"""
        self.assertTrue(self.mono_tool_py.exists(), 
                       f"mono_tool.pyが存在しません: {self.mono_tool_py}")

    def test_files_match(self):
        """mono_tool.pyとmono_tool_backup.pyの内容が一致することを確認"""
        if not self.mono_tool_py.exists():
            self.skipTest("mono_tool.pyが存在しないためスキップ")
        if not self.mono_tool_backup_py.exists():
            self.skipTest("バックアップファイルが存在しないためスキップ")
        
        # 両ファイルの内容を読み込み
        with open(self.mono_tool_py, 'r', encoding='utf-8') as f:
            current_content = f.read().strip()
        
        with open(self.mono_tool_backup_py, 'r', encoding='utf-8') as f:
            backup_content = f.read().strip()
        
        # 完璧な一致を確認
        self.assertEqual(current_content, backup_content, 
                        "mono_tool.pyとバックアップファイルの内容が一致しません。")


@tool
def test_mono() -> str:
    """
    mono_tool.pyとバックアップファイルの比較テストを実行します。
    
    Returns:
        テスト実行結果のメッセージ
    """
    try:
        # 現在のファイルのディレクトリでテストを実行
        current_dir = Path(__file__).parent.parent.parent  # craft-aiディレクトリ
        
        # テストクラスを直接インポートして実行する
        import sys
        import os
        
        # カレントディレクトリをPythonパスに追加
        sys.path.insert(0, str(current_dir))
        
        # テストスイートを作成
        import unittest
        from tools.test_mono_tool.tool import TestMonoToolComparison
        
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestMonoToolComparison))
        
        # テスト結果を取得するためのテキストランナーを作成
        from io import StringIO
        output = StringIO()
        runner = unittest.TextTestRunner(stream=output, verbosity=2)
        result = runner.run(suite)
        
        # 結果を整形
        output_lines = []
        
        if result.wasSuccessful():
            return "OK テスト成功\n\nすべてのテストケースが通過しました。mono_tool.pyとバックアップファイルの整合性が確認されました。"
        else:
            # 失敗の詳細を含めて返す
            failures = []
            for test, traceback in result.failures + result.errors:
                if 'AssertionError:' in traceback:
                    error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
                else:
                    error_msg = 'エラーが発生しました'
                failures.append(f"- {test}: {error_msg}")
            
            return f"ERROR テスト失敗\n\n{len(result.failures + result.errors)}個のテストが失敗しました:\n" + "\n".join(failures)
        
    except Exception as e:
        return f"ERROR テスト実行エラー: {str(e)}"

