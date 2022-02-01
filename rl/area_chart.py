def spec(x, color):
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "data"},
        "height": 400,
        "width": 600,
        "mark": "area",
        "encoding": {
            "x": {"field": x},
            "y": {
                "aggregate": "count",
                "field": "count",
                "axis": None,
                "stack": "normalize",
            },
            "color": {"field": color, "scale": {}},
        },
    }
