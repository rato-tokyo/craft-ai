# Craft AI ツール作成ガイド

## Craft AIとは

Craft AIは、フラグベース循環システムによるプラグイン型AIチャットアプリケーションです。AIアシスタントと対話し、様々なツールを使って機能を拡張できます。

## ツールとは

ツールとは、AIアシスタントが実行できる関数のことです。Craft AIでは、LangChainの`@tool`デコレータを使用してツール関数を定義します。ツールを使うことで、AIアシスタントはシステムコマンドの実行やファイル操作などの機能を利用できるようになります。

## ツール作成の基本

### 1. ディレクトリ構造

新しいツールを作成するには、`tools`ディレクトリに新しいディレクトリを作成し、その中に`tool.py`ファイルを配置します。

```
tools/
  ├── your_tool_name/
  │   └── tool.py
```

### 2. ツールクラスの定義

`tool.py`ファイルには、`Tool`クラスを定義する必要があります。このクラスはツールの設定情報を保持し、ツール関数からアクセスできるようにします。

```python
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
```

### 3. ツール関数の定義

ツール関数は、`@tool`デコレータを使用して定義します。関数名はスネークケースで、関数の説明はdocstringに記述します。

```python
from langchain_core.tools import tool

@tool
def your_tool_function(param1: str, param2: int) -> str:
    """
    ツール関数の説明。AIアシスタントに表示されます。
    
    Args:
        param1: パラメータ1の説明
        param2: パラメータ2の説明
        
    Returns:
        戻り値の説明
    """
    # 処理内容
    return "結果"
```

### 4. 設定ファイルへの登録

作成したツールを使用するには、`settings.json`ファイルの`tools`配列に追加します。

```json
"tools": [
  {
    "path": "tools/your_tool_name/tool.py",
    "settings": {
      "option1": "value1",
      "option2": "value2"
    }
  }
]
```

## ツール作成のベストプラクティス

1. **明確な関数名と説明**: ツール関数の名前と説明は、AIアシスタントが適切に使用できるよう明確にします。
2. **型アノテーション**: パラメータと戻り値には型アノテーションを付けて、AIアシスタントが正しい型の値を渡せるようにします。
3. **エラーハンドリング**: ツール関数内でエラーが発生した場合は、適切なエラーメッセージを返します。
4. **AppDataの活用**: ツールから会話履歴やフラグなどのアプリケーションデータにアクセスする場合は、`app_data`を使用します。

## 動作の仕組み

Craft AIは起動時に`_load_tools`メソッドで設定ファイルに記載されたツールを読み込み、`@tool`デコレータが付いた関数を収集します。AIアシスタントはこれらのツールを必要に応じて呼び出します。

## まとめ

このガイドに従ってツールを作成することで、Craft AIの機能を拡張できます。 