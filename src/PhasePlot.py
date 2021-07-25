import numpy as np

class PhasePlot:

    def __init__(self, title):
        self.ax = None
        self.title = title
        self.frames = []
        self.legend = []
        self.colors = []

    def reset_dimensions(self):
        self.ax.vlines(0,1, np.max([df['total'].max() for df in self.frames])*1.2, linestyles='dashed', color="black")
        err_max = np.max([ np.max(np.abs(df[df['err'].isna() == False]['err'])) for df in self.frames])*1.1
        self.err_max = err_max
        self.ax.set_xbound(-100, 100)
        self.annotations_x = err_max*1.1
        self.annotations_x_delta = err_max*2/100
        self.annotations_y = np.max([ np.max(np.abs(df[df['total'].isna() == False]['total'])) for df in self.frames])*0.8
        self.annotations_y_delta = np.log(self.annotations_y)/30

    def add(self, df, offset=0, color=None, legend=None):
        if color==None:
            color=f"C{len(self.colors)}"
        slice = df[df.index>=offset].copy()
        slice.loc[slice['total']==0,'total']=1
        if len(self.frames) == 0:
            self.ax = slice.plot(x='err', y='total', figsize=(16,10), color=color, title=self.title)
            self.ax.set_yscale('log')
            self.ax.set_xlabel("Modeling Error %")
            self.ax.set_ylabel("New Cases")
        else:
            self.ax.plot(slice['err'], slice['total'], color=color)

        self.frames.append(slice)
        self.colors.append(color)
        self.reset_dimensions()
        self.legend.append(legend)
        self.ax.legend(self.legend)
        self.ax.scatter(x=slice.err, y=slice.total, color=color)
        return len(self.frames)-1

    def add_horizon(self, horizon_df, legend, color="black", linestyle="dashed"):
        self.ax.plot(horizon_df["err"], horizon_df["total"], linestyle=linestyle, color=color)
        self.frames.append(horizon_df)
        self.colors.append(color)
        self.legend.append(legend)
        self.ax.legend(self.legend)
        self.reset_dimensions()
        return len(self.frames)-1

    def add_label(self, index, date, text):
        df = self.frames[index]

        rec = df.loc[df["date"]==date, ["err", "total", "cumulative"]]
        err, total, cumulative = rec.values[0]
        self.add_xy_label(
            index,
            err,
            total,
            f"{date} - day:{rec.index.values[0]}, new:{int(total)}, cases:{int(cumulative)} - {text}")
        return

    def add_xy_label(self, index, err, total, text):
        self.ax.text(
            self.annotations_x,
            self.annotations_y,
            text,
            color=self.colors[index]
        )
        x=[self.annotations_x-self.annotations_x_delta*2, err]
        y=[self.annotations_y, total]
        self.ax.plot(x, y, color="black", linestyle="dotted")
        self.ax.scatter(x=[err], y=[total], color="black")
        self.annotations_y = np.exp(np.log(self.annotations_y) - self.annotations_y_delta)
        return
