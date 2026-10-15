---
name: repo-rich
description: Map of the Textualize/rich repository (the Rich terminal rendering library (styles, markup, tables, progress, syntax, console export)). Load it first when the task's repository is Textualize/rich; it says which file holds each class or function so you can go straight to read_file.
---

Repository `Textualize/rich`: the Rich terminal rendering library (styles, markup, tables, progress, syntax, console export). Source package: `rich/`. Tests: `tests/` (63 `test_*.py` files at the top level).

Use this map to pick the file, then confirm with one scoped `git grep -n "<symbol>" rich/` and `read_file` on that file only.
Do not grep the whole repository, docs or examples. The map was taken from one snapshot: when a name is missing, the code moved; fall back to `git grep`.

Files most often changed by fixes in this repository (count over 48 reference fixes): `rich/console.py` (7), `rich/cells.py` (6), `rich/segment.py` (5), `rich/markdown.py` (5), `rich/syntax.py` (4), `rich/default_styles.py` (3), `rich/table.py` (3), `rich/traceback.py` (3), `rich/text.py` (3), `rich/repr.py` (2), `rich/ansi.py` (2), `rich/prompt.py` (2).
Tests most often extended by those fixes: `tests/test_console.py`, `tests/test_cells.py`, `tests/test_text.py`, `tests/test_markdown.py`, `tests/test_syntax.py`, `tests/test_segment.py`.
Focused test run: `python -m pytest tests/test_<module>.py -q -x 2>&1 | tail -20` (the test file is usually named after the module).

Module map (top-level classes with their methods, public functions):
- `rich/__init__.py`: get_console(), reconfigure(), print(), print_json(), inspect()
- `rich/__main__.py`: class ColorBox(__rich_console__), make_test_card()
- `rich/_extension.py`: load_ipython_extension()
- `rich/_fileno.py`: get_fileno()
- `rich/_inspect.py`: class Inspect(__init__, _make_title, _get_signature, _render, _get_formatted_doc), get_object_types_mro(), get_object_types_mro_as_strings(), is_object_one_of_types()
- `rich/_log_render.py`: class LogRender(__init__, __call__)
- `rich/_loop.py`: loop_first(), loop_last(), loop_first_last()
- `rich/_null_file.py`: class NullFile(close, isatty, read, readable, readline, readlines, ...)
- `rich/_pick.py`: pick_bool()
- `rich/_ratio.py`: class Edge, ratio_resolve(), ratio_reduce(), ratio_distribute()
- `rich/_stack.py`: class Stack(top, push)
- `rich/_timer.py`: timer()
- `rich/_win32_console.py`: class LegacyWindowsError, class WindowsCoordinates(from_param), class CONSOLE_SCREEN_BUFFER_INFO, class CONSOLE_CURSOR_INFO, GetStdHandle(), GetConsoleMode(), FillConsoleOutputCharacter(), FillConsoleOutputAttribute(), SetConsoleTextAttribute(), GetConsoleScreenBufferInfo(), SetConsoleCursorPosition(), GetConsoleCursorInfo(), SetConsoleCursorInfo(), SetConsoleTitle() +1 more
- `rich/_windows.py`: class WindowsConsoleFeatures
- `rich/_windows_renderer.py`: legacy_windows_render()
- `rich/_wrap.py`: words(), divide_line()
- `rich/abc.py`: class RichRenderable
- `rich/align.py`: class Align(__init__, left, center, right, __rich_console__), class VerticalCenter(__init__, __rich_console__)
- `rich/ansi.py`: class _AnsiToken, class AnsiDecoder(__init__, decode, decode_line)
- `rich/bar.py`: class Bar(__init__, __rich_console__)
- `rich/box.py`: class Box(__init__, substitute, get_plain_headed_box, get_top, get_row, get_bottom)
- `rich/cells.py`: class CellTable, get_character_cell_size(), cached_cell_len(), cell_len(), split_graphemes(), split_text(), set_cell_size(), chop_cells()
- `rich/color.py`: class ColorSystem, class ColorType, class ColorParseError, class Color(system, is_system_defined, is_default, get_truecolor, from_ansi, from_triplet, ...), parse_rgb_hex(), blend_rgb()
- `rich/color_triplet.py`: class ColorTriplet(hex, rgb, normalized)
- `rich/columns.py`: class Columns(__init__, add_renderable, __rich_console__)
- `rich/console.py`: class NoChange, class ConsoleDimensions, class ConsoleOptions(ascii_only, copy, update, update_width, update_height, reset_height, ...), class RichCast, class ConsoleRenderable(__rich_console__), class CaptureError, class NewLine(__init__, __rich_console__), class ScreenUpdate(__init__, __rich_console__), class Capture(__init__, get), class ThemeContext(__init__), class PagerContext(__init__), class ScreenContext(__init__, update), class Group(__init__, renderables, __rich_console__), group() +5 more
- `rich/constrain.py`: class Constrain(__init__, __rich_console__)
- `rich/containers.py`: class Renderables(__init__, __rich_console__, append), class Lines(__init__, __rich_console__, append, extend, pop, justify)
- `rich/control.py`: class Control(__init__, bell, home, move, move_to_column, move_to, ...), strip_control_codes(), escape_control_codes()
- `rich/diagnose.py`: report()
- `rich/emoji.py`: class NoEmoji, class Emoji(__init__, replace, __rich_console__)
- `rich/errors.py`: class ConsoleError, class StyleError, class StyleSyntaxError, class MissingStyle, class StyleStackError, class NotRenderableError, class MarkupError, class LiveError, class NoAltScreen
- `rich/file_proxy.py`: class FileProxy(__init__, rich_proxied_file, write, flush, fileno, isatty)
- `rich/filesize.py`: pick_unit_and_suffix(), decimal()
- `rich/highlighter.py`: class Highlighter(__call__, highlight), class NullHighlighter(highlight), class RegexHighlighter(highlight), class ReprHighlighter, class JSONHighlighter(highlight), class ISO8601Highlighter
- `rich/json.py`: class JSON(__init__, from_data)
- `rich/jupyter.py`: class JupyterRenderable(__init__, _repr_mimebundle_), class JupyterMixin(_repr_mimebundle_), display(), print()
- `rich/layout.py`: class LayoutRender, class LayoutError, class NoSplitter, class _Placeholder(__init__, __rich_console__), class Splitter(get_tree_icon, divide), class RowSplitter(get_tree_icon, divide), class ColumnSplitter(get_tree_icon, divide), class Layout(__init__, renderable, children, map, get, tree, ...)
- `rich/live.py`: class _RefreshThread(__init__, stop, run), class Live(__init__, is_started, get_renderable, start, stop, _enable_redirect_io, ...)
- `rich/live_render.py`: class LiveRender(__init__, last_render_height, set_renderable, position_cursor, restore_cursor, __rich_console__)
- `rich/logging.py`: class RichHandler(__init__, get_level_text, emit, render_message, render)
- `rich/markdown.py`: class MarkdownElement(create, on_enter, on_text, on_leave, on_child_close, __rich_console__), class UnknownElement, class TextElement(on_enter, on_text, on_leave), class Paragraph(create, __init__, __rich_console__), class HeadingFormat, class Heading(create, on_enter, __init__, __rich_console__), class CodeBlock(create, __init__, __rich_console__), class BlockQuote(__init__, on_child_close, __rich_console__), class HorizontalRule(__rich_console__), class TableElement(__init__, on_child_close, __rich_console__), class TableHeaderElement(__init__, on_child_close), class TableBodyElement(__init__, on_child_close), class TableRowElement(__init__, on_child_close), class TableDataElement(create, __init__, on_text) +6 more
- `rich/markup.py`: class Tag(markup), escape(), render()
- `rich/measure.py`: class Measurement(span, normalize, with_maximum, with_minimum, clamp, get), measure_renderables()
- `rich/padding.py`: class Padding(__init__, indent, unpack, __rich_console__)
- `rich/pager.py`: class Pager(show), class SystemPager(_pager, show)
- `rich/palette.py`: class Palette(__init__, match)
- `rich/panel.py`: class Panel(__init__, fit, _title, _subtitle, __rich_console__)
- `rich/pretty.py`: install(), class Pretty(__init__, __rich_console__), is_expandable(), class Node(iter_tokens, check_length, render), class _Line(expandable, check_length, expand), traverse(), pretty_repr(), pprint()
- `rich/progress.py`: class _TrackThread(__init__, run), track(), class _Reader(__init__, closed, fileno, isatty, mode, name, ...), class _ReadContext(__init__), wrap_file(), open(), open(), open(), class ProgressColumn(__init__, get_table_column, __call__, render), class RenderableColumn(__init__, render), class SpinnerColumn(__init__, set_spinner, render), class TextColumn(__init__, render), class BarColumn(__init__, render), class TimeElapsedColumn(render) +10 more
- `rich/progress_bar.py`: class ProgressBar(__init__, percentage_completed, _get_pulse_segments, update, _render_pulse, __rich_console__)
- `rich/prompt.py`: class PromptError, class InvalidResponse(__init__), class PromptBase(__init__, ask, ask, ask, render_default, make_prompt, ...), class Prompt, class IntPrompt, class FloatPrompt, class Confirm(render_default, process_response)
- `rich/protocol.py`: is_renderable(), rich_cast()
- `rich/region.py`: class Region
- `rich/repr.py`: class ReprError, auto(), auto(), auto(), rich_repr(), rich_repr(), rich_repr()
- `rich/rule.py`: class Rule(__init__, __rich_console__, _rule_line)
- `rich/scope.py`: render_scope()
- `rich/screen.py`: class Screen(__init__, __rich_console__)
- `rich/segment.py`: class ControlType, class Segment(cell_length, is_control, _split_cells, split_cells, line, apply_style, ...), class Segments(__init__, __rich_console__), class SegmentLines(__init__, __rich_console__)
- `rich/spinner.py`: class Spinner(__init__, __rich_console__, render, update)
- `rich/status.py`: class Status(__init__, renderable, console, update, start, stop)
- `rich/style.py`: class _Bit(__init__), class Style(__init__, null, from_color, from_meta, on, link_id, ...), class StyleStack(__init__, current, push, pop)
- `rich/styled.py`: class Styled(__init__, __rich_console__)
- `rich/syntax.py`: class SyntaxTheme(get_style_for_token, get_background_style), class PygmentsSyntaxTheme(__init__, get_style_for_token, get_background_style), class ANSISyntaxTheme(__init__, get_style_for_token, get_background_style), class _SyntaxHighlightRange, class PaddingProperty, class Syntax(get_theme, __init__, from_path, guess_lexer, _get_base_style, _get_token_color, ...)
- `rich/table.py`: class Column(copy, cells, flexible), class Row, class _Cell, class Table(__init__, grid, expand, expand, _extra_width, row_count, ...)
- `rich/terminal_theme.py`: class TerminalTheme(__init__)
- `rich/text.py`: class Span(split, move, right_crop, extend), class Text(__init__, cell_len, markup, from_markup, from_ansi, styled, ...)
- `rich/theme.py`: class Theme(__init__, config, from_file, read), class ThemeStackError, class ThemeStack(__init__, push_theme, pop_theme)
- `rich/traceback.py`: install(), class Frame, class _SyntaxError, class Stack, class Trace, class PathHighlighter, class Traceback(__init__, from_exception, extract, __rich_console__, _render_syntax_error, _guess_lexer, ...)
- `rich/tree.py`: class Tree(__init__, add, __rich_console__)
- `rich/_unicode_data/__init__.py`: load()
