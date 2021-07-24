class GrowthPlot:
    def __init__(self, date):
        self.date = date
        self.frames = []
        self.colors = []
        self.legend = []
        pass

    def add(self, df, offset=0, color=None, legend=None):
        if color==None:
            color=f"C{len(self.colors)}"
        slice = df[df.index>=offset].copy()
        if len(self.frames) == 0:
            self.ax = slice['ols-growth-rate'].plot(title=f"Daily Cumulative Growth % ({self.date})", figsize=(10,10), color=color)
            self.ax.set_xlabel("Days Since 1st Case")
            self.ax.set_ylabel("Daily Cumulative Growth %")
            self.ax.set_yticks(range(0,110,10))
#            self.ax.text(80,101, "Based on OLS regression of observed 5-day growth")
            self.ax.grid()
        else:
            self.ax.plot(slice['ols-growth-rate'], color=color)

        self.frames.append(slice)
        self.colors.append(color)
        self.reset_dimensions()
        self.legend.append(legend)
        self.ax.legend(self.legend)
        return len(self.frames)-1

    def reset_dimensions(self):
        pass
