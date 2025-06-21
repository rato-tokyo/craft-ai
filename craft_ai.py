#!/usr/bin/env python3
"""
Craft AI - フラグベース循環システムによるプラグイン型AIチャットアプリケーション

ドキュメント仕様に基づく定義のみの実装
- craft_ai_design_and_lifecycle.md に記載された仕様に準拠
- 実装は含まず、クラス・メソッド定義のみ
"""

import json
import importlib.util
import os
import sys
import time
from typing import Dict, List, Any, Optional, Iterator
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field

# 必須ライブラリのimport（fail fast）
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage, AIMessageChunk
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


class CustomChatAnthropic(BaseChatModel):
    """高速化対応のカスタムChatAnthropicモデル"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-latest", max_tokens: int = 4000, **kwargs):
        super().__init__(**kwargs)
        
        # 環境変数からAPIキーを取得（fail fast）
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("環境変数ANTHROPIC_API_KEYが設定されていません")
        
        # 内部でChatAnthropicを使用
        self._internal_llm = ChatAnthropic(
            model=model, 
            max_tokens=max_tokens,
            anthropic_api_key=api_key
        )
        self._model_name = model
        self._max_tokens = max_tokens
    
    @property
    def _llm_type(self) -> str:
        """モデルタイプを返す"""
        return "custom_chat_anthropic"
    
    def bind_tools(self, tools: List) -> "CustomChatAnthropic":
        """ツールをバインドする"""
        if tools:
            self._internal_llm = self._internal_llm.bind_tools(tools)
        return self
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """メッセージを送信してAI応答を取得"""
        return self._internal_llm.invoke(messages, **kwargs)
    
    def stream(self, messages: List[BaseMessage], **kwargs):
        """ストリーミングでメッセージを送信"""
        return self._internal_llm.stream(messages, **kwargs)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同期的にチャット応答を生成"""
        # 内部LLMに処理を委譲
        result = self._internal_llm._generate(messages, stop, run_manager, **kwargs)
        return result


class AppData:
    """アプリケーションデータ管理（辞書形式アクセス）- シングルトンパターン"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_file: str = None):
        """シングルトンインスタンスを返す"""
        if cls._instance is None:
            cls._instance = super(AppData, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: str = None):
        """JSONファイルから初期化（初回のみ）"""
        if self._initialized:
            return
        
        if config_file is None:
            raise ValueError("初回初期化時にはconfig_fileが必要です")
        
        # .envファイルの読み込み
        load_dotenv()
        
        with open(config_file, encoding="utf-8") as f:
            settings = json.load(f)
        
        # 設定値を使用して各コンポーネントを初期化
        self._load_flags(settings)
        self._load_plugins(settings)
        self._load_tools(settings)
        self._load_inputs(settings)
        self._load_messages(settings)
        
        # AI初期化（fail fast）
        model = settings.get("model", "claude-3-5-sonnet-latest")
        
        # すべてのツール関数を収集
        all_tool_functions = []
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'functions'):
                all_tool_functions.extend(tool.functions)
        
        AI.initialize_from_settings(model, all_tool_functions, self.system_prompt, self)
        
        # 初期化完了フラグ
        self._initialized = True
    
    def __getitem__(self, key: str) -> Any:
        """辞書形式でのアクセスを可能にする"""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """辞書形式での設定を可能にする"""
        setattr(self, key, value)
    
    def _load_flags(self, settings: Dict[str, Any]) -> None:
        """フラグ読み込み"""
        self.flags = settings["flags"]
    
    def _load_plugins(self, settings: Dict[str, Any]) -> None:
        """プラグイン読み込み"""
        self.plugins = {}  # プラグイン名をキーとしてプラグインインスタンスを保存
        
        for plugin_config in settings["plugins"]:
            plugin_path = plugin_config["path"]
            plugin_settings = plugin_config.get("settings", {})
            
            # プラグイン名を取得（パスからディレクトリ名を抽出）
            plugin_name = os.path.basename(os.path.dirname(plugin_path))
            
            # モジュールの読み込み
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            # プラグインインスタンス作成（設定情報とAppDataへの参照を渡す）
            plugin = plugin_module.Plugin(self, plugin_settings)
            
            # プラグインをディクショナリに追加
            self.plugins[plugin_name] = plugin
    
    def _load_tools(self, settings: Dict[str, Any]) -> None:
        """ツール読み込み"""
        self.tools = {}  # ツール名をキーとしてツールインスタンスを保存
        
        for tool_config in settings["tools"]:
            tool_path = tool_config["path"]
            tool_settings = tool_config.get("settings", {})
            
            # ツール名を取得（パスからディレクトリ名を抽出）
            tool_name = os.path.basename(os.path.dirname(tool_path))
            
            # モジュールの読み込み
            spec = importlib.util.spec_from_file_location("tool", tool_path)
            tool_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_module)
            
            # ツールインスタンス作成（設定情報とAppDataへの参照を渡す）
            tool = tool_module.Tool(settings=tool_settings, app_data=self)
            
            # ツール関数をインスタンスに関連付け
            tool.functions = []
            for name in dir(tool_module):
                obj = getattr(tool_module, name)
                if hasattr(obj, '__call__') and hasattr(obj, 'name'):
                    # LangChainツール関数をツールインスタンスに追加
                    tool.functions.append(obj)
            
            # ツールをディクショナリに追加
            self.tools[tool_name] = tool
    
    def _load_inputs(self, settings: Dict[str, Any]) -> None:
        """入力配列初期化"""
        self.inputs = settings["inputs"]
    
    def _load_messages(self, settings: Dict[str, Any]) -> None:
        """会話履歴初期化"""
        # システムプロンプトを空で初期化
        self.system_prompt = ""
        
        # ツール情報をシステムプロンプトに追加
        if hasattr(self, "tools") and self.tools:
            tool_info = "利用可能なツール:\n"
            for tool_name, tool in self.tools.items():
                if hasattr(tool, 'functions'):
                    for func in tool.functions:
                        if hasattr(func, 'description'):
                            tool_info += f"- {func.name}: {func.description}\n"
                        elif hasattr(func, '__doc__') and func.__doc__:
                            tool_info += f"- {func.name}: {func.__doc__.strip()}\n"
                        else:
                            tool_info += f"- {func.name}\n"
            self.system_prompt = tool_info
        
        # 常にシステムメッセージをmessages[0]に配置
        self.messages = [SystemMessage(content=self.system_prompt)]
    
    def _validate_system_message_position(self) -> None:
        """システムメッセージの位置を検証"""
        # messages[0]がSystemMessageでない場合は例外
        if len(self.messages) == 0 or not isinstance(self.messages[0], SystemMessage):
            raise RuntimeError("messages[0]にSystemMessageが存在しません")
        
        # messages[1]以降にSystemMessageがある場合は例外
        for i, message in enumerate(self.messages[1:], 1):
            if isinstance(message, SystemMessage):
                raise RuntimeError(f"messages[{i}]にSystemMessageが存在します。SystemMessageはmessages[0]にのみ配置可能です")
    
    def add_message(self, message: BaseMessage) -> None:
        """メッセージを追加（位置検証付き）"""
        # SystemMessageの場合は位置チェック
        if isinstance(message, SystemMessage):
            if len(self.messages) > 0:
                # 既存のSystemMessage（messages[0]）を更新
                self.messages[0] = message
            else:
                # 初回のSystemMessage追加
                self.messages.append(message)
        else:
            # SystemMessage以外は通常通り追加
            self.messages.append(message)
        
        # 追加後に位置検証
        self._validate_system_message_position()


class AI:
    """AIグローバル変数クラス"""
    
    _agent = None
    _initialized = False
    _app_data = None
    
    @classmethod
    def _initialize(cls, model: str, system_prompt: str, app_data: AppData, tools: List = None) -> None:
        """AIエージェントの初期化"""
        if cls._initialized:
            return
        
        # AppDataの参照を保存
        cls._app_data = app_data
        
        # CustomChatAnthropicの初期化（fail fast）
        llm = CustomChatAnthropic(model=model, max_tokens=4000)
        
        # ツールのバインド
        valid_tools = []
        if tools:
            for tool in tools:
                if hasattr(tool, '__call__') and hasattr(tool, 'name'):
                    valid_tools.append(tool)
            
            if valid_tools:
                llm = llm.bind_tools(valid_tools)
        
        # ストリーミング対応LLMノードの定義
        def llm_node(state: MessagesState):
            """LLMノード: メッセージを受け取り、AIの応答を返す（ストリーミング対応）"""
            messages = state["messages"]
            
            # 直接的な応答生成（ストリーミングは後で実装）
            response = llm.invoke(messages)
            return {"messages": [response]}
        
        # グラフの構築
        workflow = StateGraph(MessagesState)
        workflow.add_node("llm", llm_node)
        
        # ツールノードの追加（有効なツールがある場合のみ）
        if valid_tools:
            tool_node = ToolNode(valid_tools)
            workflow.add_node("tools", tool_node)
            # エッジの定義
            workflow.add_edge(START, "llm")
            workflow.add_conditional_edges("llm", tools_condition)
            workflow.add_edge("tools", "llm")
            workflow.add_edge("llm", END)
        else:
            workflow.add_edge(START, "llm")
            workflow.add_edge("llm", END)
        
        # グラフのコンパイル
        cls._agent = workflow.compile()
        cls._initialized = True
    
    @classmethod
    def send(cls) -> str:
        """AIにデータ送信"""
        # 送信前にSystemMessage位置を検証
        cls._app_data._validate_system_message_position()
        
        # LangGraphエージェントを実行
        result = cls._agent.invoke({"messages": cls._app_data.messages})
            
        # 結果のメッセージを取得してAppDataに保存
        cls._app_data.messages = result["messages"]
        
        # 受信後にSystemMessage位置を検証
        cls._app_data._validate_system_message_position()
        
        # 最後のAIメッセージの内容を返す
        for msg in reversed(cls._app_data.messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return ""
    
    @classmethod
    def initialize_from_settings(cls, model: str, tools: List = None, system_prompt: str = "", app_data: AppData = None) -> None:
        """設定からAIを初期化"""
        cls._initialize(model, system_prompt, app_data, tools)


def get_user_input() -> str:
    """ユーザー入力取得"""
    return input("> ")


def execute_plugins_until_ready_for_ai(app_data: AppData) -> None:
    """ready_for_ai = trueになるまでプラグインを順次実行"""
    # プラグインが無効の場合は即座にready_for_aiをtrueに設定
    if not app_data["plugins"]:
        app_data["flags"]["ready_for_ai"] = True
        return
    
    # 既にready_for_aiがtrueの場合は何もしない
    if app_data["flags"]["ready_for_ai"]:
        return
    
    max_iterations = 100  # 無限ループ防止
    
    for i in range(max_iterations):
        # 全プラグインのon_flag()実行
        for plugin_name, plugin in app_data["plugins"].items():
            plugin.on_flag(app_data)
        
        # ready_for_aiフラグチェック
        if app_data["flags"]["ready_for_ai"]:
            break
    else:
        raise RuntimeError("プラグインループ最大回数超過: ready_for_ai")


def execute_plugins_until_ready_for_user(app_data: AppData) -> None:
    """ready_for_user = trueになるまでプラグインを順次実行"""
    # プラグインが無効の場合は即座にready_for_userをtrueに設定
    if not app_data["plugins"]:
        app_data["flags"]["ready_for_user"] = True
        return
    
    # 既にready_for_userがtrueの場合は何もしない
    if app_data["flags"]["ready_for_user"]:
        return
    
    max_iterations = 100  # 無限ループ防止
    
    for i in range(max_iterations):
        # 全プラグインのon_flag()実行
        for plugin_name, plugin in app_data["plugins"].items():
            plugin.on_flag(app_data)
        
        # ready_for_userフラグチェック
        if app_data["flags"]["ready_for_user"]:
            break
    else:
        raise RuntimeError("プラグインループ最大回数超過: ready_for_user")


def close_plugins(app_data: AppData) -> None:
    """プラグインの終了処理"""
    for plugin_name, plugin in app_data["plugins"].items():
        if hasattr(plugin, 'cleanup'):
            plugin.cleanup()


def main() -> None:
    """Craft AI メインライフサイクル"""
    try:
        # === 初期化フェーズ ===
        app_data = AppData("settings.json")
        
        # === メインループ ===
        while True:
            # フラグの初期化
            app_data["flags"]["ready_for_ai"] = False
            app_data["flags"]["ready_for_user"] = False
            
            # inputs配列の処理
            if len(app_data["inputs"]) > 0:
                # inputs配列から最初の入力を取得
                user_input = app_data["inputs"].pop(0)
            else:
                # inputs配列が空の場合はユーザー入力を求める
                user_input = get_user_input()
            
            # exitコマンドの処理（inputs配列からの場合も即座に終了）
            if user_input.lower() == "exit":
                close_plugins(app_data)
                return
                
            # ユーザー入力をメッセージに追加
            app_data.add_message(HumanMessage(content=user_input))
            
            # ユーザー入力後フラグ設定。自動でafter_ai_response=falseになる仕様
            app_data["flags"]["after_user_input"] = True
            app_data["flags"]["after_ai_response"] = False
            
            # ready_for_ai = trueになるまでプラグインの順次実行をループする
            execute_plugins_until_ready_for_ai(app_data)
            
            # AIにデータ送信。AIはAppDataには含まれないグローバル変数
            ai_response = AI.send()
            
            # AI応答後フラグ設定。自動でafter_user_input=falseになる仕様
            app_data["flags"]["after_ai_response"] = True
            app_data["flags"]["after_user_input"] = False
            
            # ready_for_user = trueになるまでプラグインの順次実行をループする
            execute_plugins_until_ready_for_user(app_data)
    
    except KeyboardInterrupt:
        # Ctrl+C検出時の終了処理
        try:
            close_plugins(app_data)
        except NameError:
            # app_dataが初期化されていない場合
            pass
    
    finally:
        pass


if __name__ == "__main__":
    main() 