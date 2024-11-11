import fasthtml.common as fh

main_css = fh.Link(rel="stylesheet", href="main.css", type="text/css")

app = fh.FastHTML(hdrs=(main_css))
rt = app.route


@rt("/{fname:path}.{ext:static}")
def get(fname: str, ext: str):
    return fh.FileResponse(f"static/{fname}.{ext}")


@rt("/")
def get():
    return fh.Div(fh.H1("Hello World!", cls="multicolor-text text-[50px]"))


fh.serve()
