from urllib.parse import urlparse, parse_qsl, unquote_plus

# scheme 0 URL scheme specifier
# netloc 1 Network location part
# path 2 Hierarchical path
# params 3 Parameters for last path element
# query 4 Query component
# fragment 5  Fragment identifier
url1 = input()
url2 = input()


class Url(object):

    def __init__(self, url):
        parts = urlparse(url)
        _query = frozenset(parse_qsl(parts.query))
        _path = unquote_plus(parts.path)
        parts = parts._replace(query=_query, path=_path)
        self.parts = parts

    def __eq__(self, other):
        return self.parts == other.parts

    def __hash__(self):
        return hash(self.parts)


if Url(url1).parts.scheme != Url(url2).parts.scheme:
    print("URL schemes are different: ")
    print("scheme of first url: " + Url(url1).parts.scheme)
    print("scheme of second url: " + Url(url2).parts.scheme)
elif Url(url1).parts.netloc != Url(url2).parts.netloc:
    print("URL netlocs are different: ")
    print("netloc of first url: " + Url(url1).parts.netloc)
    print("netloc of second url: " + Url(url2).parts.netloc)
elif Url(url1).parts.path != Url(url2).parts.path:
    print("URL paths are different: ")
    print("paths of first url: " + Url(url1).parts.path)
    print("paths of second url: " + Url(url2).parts.path)
elif Url(url1).parts.query != Url(url2).parts.query:
    print("URL query are different: ")
    print("query of first url: " + Url(url1).parts.query)
    print("query of second url: " + Url(url2).parts.query)
elif Url(url1).parts.fragment != Url(url2).parts.fragment:
    print("URL fragments are different: ")
    print("fragment of first url: " + Url(url1).parts.fragment)
    print("fragment of second url: " + Url(url2).parts.fragment)
else:
    print("URL are similar")
