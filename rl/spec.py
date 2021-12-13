def spec(x, y, visualizer_url=None, color="run ID"):
    def subfigure(subfigure_params, x_kwargs, y_kwargs):
        return {
            "height": 400,
            "width": 600,
            "encoding": {
                "x": {"type": "quantitative", "field": x, **x_kwargs},
                "y": {"type": "quantitative", "field": y, **y_kwargs},
                "color": {"type": "nominal", "field": color},
                "href": {"field": "url", "type": "nominal"},
                "opacity": {
                    "value": 0.1,
                    "condition": {
                        "test": {
                            "and": [
                                {"param": "legend_selection"},
                                {"param": "hover"},
                            ]
                        },
                        "value": 1,
                    },
                },
            },
            "layer": [
                {
                    "mark": "line",
                    "params": subfigure_params,
                    **(
                        {}
                        if visualizer_url is None
                        else {
                            "transform": [
                                {
                                    "calculate": f"'{visualizer_url}' + datum['run ID']",
                                    "as": "url",
                                }
                            ],
                        }
                    ),
                }
            ],
        }

    params = [
        {
            "bind": "legend",
            "name": "legend_selection",
            "select": {
                "on": "mouseover",
                "type": "point",
                "fields": ["run ID"],
            },
        },
        {
            "bind": "legend",
            "name": "hover",
            "select": {
                "on": "mouseover",
                "type": "point",
                "fields": ["run ID"],
            },
        },
    ]

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "data"},
        "transform": [{"filter": {"field": y, "valid": True}}],
        "hconcat": [
            subfigure(
                subfigure_params=[*params, {"name": "selection", "select": "interval"}],
                x_kwargs={},
                y_kwargs={},
            ),
            subfigure(
                subfigure_params=params,
                x_kwargs={"scale": {"domain": {"param": "selection", "encoding": "x"}}},
                y_kwargs={"scale": {"domain": {"param": "selection", "encoding": "y"}}},
            ),
        ],
    }
