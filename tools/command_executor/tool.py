from langchain_core.tools import tool
import subprocess
import os

class Tool:
    """コマンド実行ツールの基底クラス"""
    
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

@tool
def execute_command(command: str) -> str:
    """
    システムコマンドを実行して結果を返します。
    
    Args:
        command (str): 実行するシステムコマンド
        
    Returns:
        str: コマンドの実行結果（標準出力、標準エラー、終了コード）
       
    """
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )
        
        output = ""
        if result.stdout:
            output += f"標準出力:\n{result.stdout}\n"
        if result.stderr:
            output += f"標準エラー:\n{result.stderr}\n"
        output += f"終了コード: {result.returncode}"
        
        return output
        
    except subprocess.TimeoutExpired:
        return "エラー: コマンドがタイムアウトしました（30秒制限）"
    except Exception as e:
        return f"エラー: コマンド実行中に予期しないエラーが発生しました: {str(e)}"
