def spec(x, y, color, history_len):
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "axes": [
            {"scale": "y", "title": y, "orient": "left"},
            {
                "scale": "x",
                "title": x,
                "orient": "bottom",
                "encode": {
                    "labels": {
                        "update": {"angle": {"value": 45}, "align": {"value": "left"}}
                    }
                },
            },
        ],
        "data": {
            "name": "data",
            "transform": [
                {"expr": f"datum['{x}']", "type": "filter"},
                {"expr": f"datum['{y}']", "type": "filter"},
                {
                    "type": "joinaggregate",
                    "fields": ["step"],
                    "ops": ["max"],
                    "as": ["maxStep"],
                },
                {
                    "type": "joinaggregate",
                    "fields": [color],
                    "ops": ["max"],
                    "as": ["maxSuccess"],
                },
                {
                    "type": "formula",
                    "as": "sliderStep",
                    "expr": " datum.maxStep * sliderValue",
                },
                {
                    "expr": f"datum['step'] >  datum.sliderStep - {history_len}",
                    "type": "filter",
                },
                {"expr": "datum['step'] <=  datum.sliderStep", "type": "filter"},
            ],
        },
        "signals": [
            {
                "name": "sliderValue",
                "value": 1,
                "bind": {"input": "range", "min": 0, "max": 1, "step": 0.01},
            }
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
                "from": {"data": "data"},
                "type": "rect",
                "encode": {
                    "enter": {
                        "x": {"field": x, "scale": "x"},
                        "y": {"field": y, "scale": "y"},
                        "width": {"band": 1, "scale": "x"},
                        "height": {"band": 1, "scale": "y"},
                        "tooltip": {"signal": f"datum['{color}']"},
                    },
                    "update": {"fill": {"field": color, "scale": "color"}},
                },
            },
            {"type": "text", "encode": {"enter": {"text": {"field": "sliderStep"}}}},
        ],
        "scales": [
            {
                "name": "x",
                "type": "band",
                "range": "width",
                "domain": {"data": "data", "field": x},
            },
            {
                "name": "y",
                "type": "band",
                "range": "height",
                "domain": {"data": "data", "field": y},
            },
            {
                "name": "color",
                "type": "linear",
                "range": {"scheme": "Viridis"},
                "domain": {"data": "data", "field": color},
                "domainMax": 1,
            },
        ],
        "height": 400,
        "width": 600,
    }
