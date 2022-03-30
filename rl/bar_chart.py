def spec(x, y, step="step"):
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "description": "A basic grouped bar chart example.",
        "width": 500,
        "height": 750,
        "padding": 5,
        "data": [
            {
                "name": "data",
                "transform": [
                    {"expr": f"datum['{y}']", "type": "filter"},
                    {"expr": f"datum['{x}']", "type": "filter"},
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
                        "as": ["maxStepPerX"],
                        "ops": ["max"],
                        "type": "joinaggregate",
                        "fields": [step],
                        "groupby": [y, "run ID"],
                    },
                    {
                        "expr": f"datum['{step}'] == datum['maxStepPerX']",
                        "type": "filter",
                    },
                    {
                        "as": "rounded",
                        "expr": f"round(100 * datum['{x}'])/ 100",
                        "type": "formula",
                    },
                ],
            }
        ],
        "signals": [
            {
                "name": "tooltip",
                "value": {},
                "on": [
                    {"events": "rect:mouseover", "update": "datum"},
                    {"events": "rect:mouseout", "update": "{}"},
                ],
            },
            {
                "bind": {"max": 1, "min": 0, step: 0.01, "input": "range"},
                "name": "sliderValue",
                "value": 1,
            },
        ],
        "scales": [
            {
                "name": "yscale",
                "type": "band",
                "domain": {"data": "data", "field": y},
                "range": "height",
                "padding": 0.2,
            },
            {
                "name": "xscale",
                "type": "linear",
                "domain": {"data": "data", "field": x},
                "range": "width",
                "round": True,
                "zero": True,
                "nice": True,
            },
            {
                "name": "color",
                "type": "ordinal",
                "domain": {"data": "data", "field": "run ID"},
                "range": {"scheme": "category20"},
            },
        ],
        "axes": [
            {
                "orient": "left",
                "scale": "yscale",
                "tickSize": 0,
                "labelPadding": 4,
                "zindex": 1,
            },
            {"orient": "bottom", "scale": "xscale"},
        ],
        "marks": [
            {
                "type": "group",
                "from": {"facet": {"data": "data", "name": "facet", "groupby": y}},
                "encode": {"enter": {"y": {"scale": "yscale", "field": y}}},
                "signals": [{"name": "height", "update": "bandwidth('yscale')"}],
                "scales": [
                    {
                        "name": "pos",
                        "type": "band",
                        "range": "height",
                        "domain": {"data": "facet", "field": "run ID"},
                    }
                ],
                "marks": [
                    {
                        "name": "bars",
                        "from": {"data": "facet"},
                        "type": "rect",
                        "encode": {
                            "enter": {
                                "y": {"scale": "pos", "field": "run ID"},
                                "height": {"scale": "pos", "band": 1},
                                "x": {"scale": "xscale", "field": x},
                                "x2": {"scale": "xscale", x: 0},
                                "fill": {"scale": "color", "field": "run ID"},
                            }
                        },
                    },
                    {
                        "type": "text",
                        "from": {"data": "bars"},
                        "encode": {
                            "enter": {
                                "x": {"field": "x2", "offset": -5},
                                "y": {
                                    "field": "y",
                                    "offset": {"field": "height", "mult": 0.5},
                                },
                                "fill": [
                                    {
                                        "test": "contrast('white', datum.fill) > contrast('black', datum.fill)",
                                        x: "white",
                                    },
                                    {x: "black"},
                                ],
                                "align": {x: "right"},
                                "baseline": {x: "middle"},
                                "text": {"field": "datum.rounded"},
                            }
                        },
                    },
                ],
            }
        ],
    }
