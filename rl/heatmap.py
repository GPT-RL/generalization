def spec(x, y, color, step="step"):
    tooltip = {"signal": "{" + f"'{y}': datum['{y}'], '{x}': datum['{x}']" + "}"}
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "axes": [
            {"scale": "y", "title": y, "orient": "left"},
            {
                "scale": "x",
                "title": x,
                "encode": {
                    "labels": {
                        "update": {"align": {"value": "left"}, "angle": {"value": 45}}
                    }
                },
                "orient": "bottom",
            },
        ],
        "data": [
            {
                "name": "data",
                "transform": [
                    {"expr": f"datum['{color}']", "type": "filter"},
                    {"expr": f"datum['{x}']", "type": "filter"},
                    {"expr": f"datum['{y}']", "type": "filter"},
                    {
                        "as": ["maxStep"],
                        "ops": ["max"],
                        "type": "joinaggregate",
                        "fields": [step],
                    },
                    {
                        "as": "sliderStep",
                        "expr": " datum.maxStep * sliderValue",
                        "type": "formula",
                    },
                    {"expr": f"datum['{step}'] <=  datum.sliderStep", "type": "filter"},
                    {
                        "type": "joinaggregate",
                        "ops": ["max"],
                        "fields": [step],
                        "groupby": [x, y],
                        "as": ["maxPairStep"],
                    },
                    {
                        "type": "filter",
                        "expr": f"datum['{step}'] == datum['maxPairStep']",
                    },
                    {
                        "type": "formula",
                        "as": "rounded",
                        "expr": f"round(100 * datum['{color}'])/ 100",
                    },
                ],
            }
        ],
        "height": 400,
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
                        "fill": {"field": color, "scale": "color"},
                        "tooltip": tooltip,
                    }
                },
            },
            {
                "type": "text",
                "from": {"data": "data"},
                "encode": {
                    "enter": {
                        "x": {"field": x, "scale": "x"},
                        "dx": {"band": 0.5, "scale": "x"},
                        "y": {"field": y, "scale": "y"},
                        "dy": {"band": 0.5, "scale": "y"},
                        "width": {"band": 1, "scale": "x"},
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "fontWeight": {"value": "bolder"},
                        "fill": {"value": "white"},
                        "height": {"band": 1, "scale": "y"},
                        "text": {"field": "rounded", "type": "quantitative"},
                        "tooltip": tooltip,
                    }
                },
            },
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
                "domainMin": 0,
            },
        ],
        "signals": [
            {
                "bind": {"max": 1, "min": 0, "step": 0.01, "input": "range"},
                "name": "sliderValue",
                "value": 1,
            }
        ],
        "width": 600,
    }
