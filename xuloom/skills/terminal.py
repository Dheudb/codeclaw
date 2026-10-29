"""Interactive $ completion backed by the same registry as the desktop."""
import re
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import has_completions
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import Lexer


class SkillCompleter(Completer):
    def __init__(self, registry): self.registry = registry

    def get_completions(self, document, complete_event):
        match = re.search(r"(?:^|\s)\$([a-zA-Z0-9-]*)$", document.text_before_cursor)
        if not match: return
        query = match.group(1)
        if query and query[0].isdigit(): return
        for skill in self.registry().active():
            if query.lower() not in (skill["name"] + " " + skill["description"]).lower(): continue
            yield Completion(skill["name"] + " ", start_position=-len(query), display="$" + skill["name"],
                             display_meta=skill["scope"] + " · " + skill["description"])


class SkillLexer(Lexer):
    def lex_document(self, document):
        def line(index):
            text = document.lines[index]
            result, cursor = [], 0
            for match in re.finditer(r"(?<!\S)\$[a-z][a-z0-9]*(?:-[a-z0-9]+)*(?=\s|$)", text):
                result.extend([("", text[cursor:match.start()]), ("class:skill", match.group())])
                cursor = match.end()
            return result + [("", text[cursor:])]
        return line


def input_session(registry, **kwargs):
    bindings = KeyBindings()
    @bindings.add("enter", filter=has_completions)
    def select(event):
        state = event.current_buffer.complete_state
        if state.current_completion is None: event.current_buffer.complete_next()
        choice = event.current_buffer.complete_state.current_completion
        event.current_buffer.apply_completion(choice)
    return PromptSession(completer=SkillCompleter(registry), complete_while_typing=True, lexer=SkillLexer(),
        key_bindings=bindings, style=Style.from_dict({"skill": "#3972c6", "completion-menu.completion.current": "bg:#dbeaff #182b43"}), **kwargs)
