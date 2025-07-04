import ast
import astunparse
import builtins

class FunctionParser(ast.NodeTransformer):
    def __init__(self, fs: dict, f_assigns: dict):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns
        self._defined_funcs = set()

        # 내장 함수 이름 목록 (예: range, len, super 등)
        self._builtin_names = set(dir(builtins))

        # 필터할 외부 util 함수 이름 (필요시 수동 추가 가능)
        self._common_known_funcs = {"Vertex", "gen_forest", "_get_grid", "_init_displays", "key_frame", "Point", "move_agent", }

    def visit_ClassDef(self, node):
        # 클래스 내부 함수 정의 수집
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._defined_funcs.add(item.name)
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)

        f_name = None
        if isinstance(node.func, ast.Name):
            f_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            f_name = node.func.attr

        if not f_name:
            return node

        # 필터링: 정의된 함수, 내장 함수, 공통 유틸 함수 제외
        if f_name in self._defined_funcs:
            return node
        if f_name in self._builtin_names:
            return node
        if f_name in self._common_known_funcs:
            return node

        f_sig = astunparse.unparse(node).strip()
        self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)

        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
                f_name = func.attr
                if f_name in self._defined_funcs or f_name in self._builtin_names or f_name in self._common_known_funcs:
                    return node
                assign_str = astunparse.unparse(node).strip()
                self._f_assigns[f_name] = assign_str
        return node