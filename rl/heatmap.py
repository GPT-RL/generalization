def spec(x, y, color):
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": 600,
        "height": 400,
        "data": {
            "name": "data",
            "transform": [
                {"type": "filter", "expr": f"datum['{x}']"},
                {"type": "filter", "expr": f"datum['{y}']"},
            ],
        },
        "scales": [
            {
                "name": "x",
                "type": "band",
                "domain": {"data": "data", "field": x},
                "range": "width",
            },
            {
                "name": "y",
                "type": "band",
                "domain": {"data": "data", "field": y},
                "range": "height",
            },
            {
                "name": "color",
                "type": "linear",
                "range": {"scheme": "Viridis"},
                "domain": {"data": "data", "field": color},
            },
        ],
        "axes": [
            {"orient": "bottom", "scale": "x", "title": x},
            {"orient": "left", "scale": "y", "title": y},
        ],
        "legends": [
            {
                "fill": "color",
                "type": "gradient",
                "title": color,
                "gradientLength": {"signal": "height - 16"},
            }
        ],
        "marks": [
            {
                "type": "rect",
                "from": {"data": "data"},
                "encode": {
                    "enter": {
                        "x": {"scale": "x", "field": x},
                        "y": {"scale": "y", "field": y},
                        "width": {"scale": "x", "band": 1},
                        "height": {"scale": "y", "band": 1},
                        "tooltip": {"signal": f"datum['{color}']"},
                    },
                    "update": {"fill": {"scale": "color", "field": color}},
                },
            }
        ],
    }
