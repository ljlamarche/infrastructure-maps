# infrastructure-maps

This is a script for creating nice visual maps of CEDAR infrastructure.

![](https://github.com/ljlamarche/infrastructure-maps/blob/main/NSF_facilities.png)

To run, modify `sites.yaml` to include the instruments/networks you are interested in and customize color and other plotting options.  Then run the script in python with `sites.yaml` (or a renamed equivilent configuration file) as a command lien option.

```
python map_facilities.py sites.yaml
```

