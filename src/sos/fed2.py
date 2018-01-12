import sys
import time

from java.awt import *
from java.io import *
from java.lang import *
from javax.swing import *
from java.util import *


from edu.mines.jtk.awt import *
from edu.mines.jtk.dsp import *
from edu.mines.jtk.io import *
from edu.mines.jtk.interp import *
from edu.mines.jtk.mosaic import *
from edu.mines.jtk.util import *
from edu.mines.jtk.util.ArrayMath import *

from sos import *

pngDir = None
pngDir = "../../png/fed2/"

seismicDir = "../../data/fed2/"
fxfile = "f3d75s"
f1,f2 = 0,0
d1,d2 = 1,1
n1,n2 = 222,440 
d1,d2 = 0.004,0.025
f1,f2 = 0.004+d1*240,0.000
s1 = Sampling(n1,d1,f1)
s2 = Sampling(n2,d2,f2)

def main(args):
  #goGaussian()
  #goSlopeVector()
  #goDiffusivity()
  goDaveLocalSmoothingFilter()
  goFastLinearDiffusion()
  #goFastNonlinearDiffusion()


# Here we show that isotropic Gaussian smoothing 
# is not good for seismic images, which is why we want 
# to construct structure-oriented (anisotropic) smoothing.
def goGaussian():
  gx = zerofloat(n1,n2)
  fx = readImage(fxfile)
  fx = gain(fx)
  rgf = RecursiveGaussianFilter(16)
  rgf.apply00(fx,gx)
  plot(fx,cmin=-1,cmax=1,cint=1.0,png="seis")
  plot(gx,cmin=-1,cmax=1,cint=1.0,png="seisGaussianSmooth")


# Here we display reflection slope vectors v=(v1,v2) estimated 
# by structure tensors. There is a relationship between the slope 
# vectors v and reflection normal vectors u=(u1,u2): v*u^T=0; we 
# therefore have v1=-u2 and v2=u1.
# The smoothing direction should be parallel to v but perpendicular to u.
def goSlopeVector():
  sig1,sig2=4,2
  fx = readImage(fxfile)
  fx = gain(fx)
  lof = LocalOrientFilter(sig1,sig2)
  u1 = zerofloat(n1,n2)
  u2 = zerofloat(n1,n2)
  el = zerofloat(n1,n2)
  lof.applyForNormalLinear(fx,u1,u2,el) 
  plot(fx,cmin=-1,cmax=1,png="seis")
  plot(fx,cmin=-2,cmax=2,v1=mul(u2,-1),v2=u1,png="slopeVector")

# Dave's structure-oriented smoothing (linear and anisotropic)
def goDaveLocalSmoothingFilter():
  fx = readImage(fxfile)
  fx = gain(fx)
  sig1,sig2=4,2
  lof = LocalOrientFilter(sig1,sig2)
  ets = lof.applyForTensors(fx)
  ets.setEigenvalues(0.001,1.0)
  sig = 56 #yields approximately equivalent smoothing strength as sig=8 in FED
  lsf = LocalSmoothingFilter()
  gx = zerofloat(n1,n2)
  lsf = LocalSmoothingFilter(0.001,200)
  start = time.time()
  lsf.apply(ets,sig,fx,gx)
  end = time.time()
  print end-start
  plot(fx,cmin=-1,cmax=1,cint=1.0,label="Amplitude",png="seis")
  plot(gx,cmin=-1,cmax=1,cint=1.0,label="Amplitude",png="daveSos")
  plot(sub(fx,gx),cmin=-0.5,cmax=0.5,cint=0.2,label="Amplitude",png="daveDiff")

# Linear and structure-oriented (anisotrpic) smoothing
def goFastLinearDiffusion():
  fx = readImage(fxfile)
  fx = gain(fx)
  sig1,sig2=4,2
  lof = LocalOrientFilter(sig1,sig2)
  ets = lof.applyForTensors(fx)
  ets.setEigenvalues(0.001,1.0)
  sig = 8 #yields approximately equivalent smoothing strength as sig=56 Dave's LSF
  cycle,limit=3,0.5
  fed = FastExplicitDiffusion()
  fed.setCycles(cycle,limit)
  start = time.time()
  gx = fed.apply(sig,ets,fx)
  end = time.time()
  print end-start
  plot(sub(fx,gx),cmin=-0.5,cmax=0.5,cint=0.2,label="Amplitude",png="fgl")
  plot(fx,cmin=-1,cmax=1,cint=1.0,label="Amplitude",png="seis")
  plot(gx,cmin=-1,cmax=1,cint=1.0,label="Amplitude",png="seisLinearSmooth")

# initial edge detection (before and after approximately 
# fault-oriented smoothing) from the original seismic image
def goDiffusivity():
  fx = readImage(fxfile)
  fx = gain(fx)
  sig1,sig2=4,2
  lof = LocalOrientFilter(sig1,sig2)
  ets = lof.applyForTensors(fx)
  ets.setEigenvalues(0.001,1.0)
  sig = 8
  lbd = 0.10
  cycle,limit=3,0.5
  fed = FastExplicitDiffusion()
  fed.setCycles(cycle,limit)
  fw = zerofloat(n1,n2)
  fws = zerofloat(n1,n2)
  fed.applyForWeightsP0(lbd,ets,fx,fw)
  fed.applyForWeightsP1(lbd,ets,fx,fws)
  plot(fx,sub(1,fw),cmin=0.3,cmax=1,cmap=jetFillExceptMin(1.0),cint=0.2,
       label="Diffusivity",png="initialDiffusivity")
  plot(fx,sub(1,fws),cmin=0.3,cmax=1,cmap=jetFillExceptMin(1.0),cint=0.2,
       label="Diffusivity",png="initialDiffusivitySmoothed")

# Linear and structure-oriented (anisotrpic) smoothing
def goFastNonlinearDiffusion():
  fx = readImage(fxfile)
  fx = gain(fx)
  sig1,sig2=4,2
  lof = LocalOrientFilter(sig1,sig2)
  ets = lof.applyForTensors(fx)
  sig = 8
  lbd = 0.10
  cycle,limit=3,0.5
  fed = FastExplicitDiffusion()
  fed.setCycles(cycle,limit)
  gx = fed.apply(sig,lbd,ets,fx)
  wp = zerofloat(n1,n2)
  ws = zerofloat(n1,n2)
  fed.applyForWeightsP(lbd,ets,gx,wp)
  fed.applyForWeightsP(lbd,ets,fx,ws)
  plot(sub(fx,gx),cmin=-0.5,cmax=0.5,cint=0.2,label="Amplitude",png="fgn")
  plot(fx,sub(1,ws),cmin=0.3,cmax=1,cmap=jetFillExceptMin(1.0),cint=0.2,
       label="Diffusivity",png="wsn")
  plot(gx,sub(1,wp),cmin=0.3,cmax=1,cmap=jetFillExceptMin(1.0),cint=0.2,
       label="Diffusivity",png="wpn")
  plot(gx,cmin=-1,cmax=1,cint=1.0,label="Amplitude",png="gxn")
  plot(fx,cmin=-1,cmax=1,cint=1.0,label="Amplitude",png="fx")
  plot(fx,t=ets,cmin=-1,cmax=1,cint=1.0,label="Amplitude",png="fxet")

def normalize(ss):
  sub(ss,min(ss),ss)
  div(ss,max(ss),ss)
  
def gain(x):
  g = mul(x,x) 
  ref = RecursiveExponentialFilter(100.0)
  ref.apply1(g,g)
  y = zerofloat(n1,n2)
  div(x,sqrt(g),y)
  return y

#############################################################################
# graphics

gray = ColorMap.GRAY
jet = ColorMap.JET
def jetFill(alpha):
  return ColorMap.setAlpha(ColorMap.JET,alpha)
def jetFillExceptMin(alpha):
  a = fillfloat(alpha,256)
  a[0] = 0.0
  return ColorMap.setAlpha(ColorMap.JET,a)
def jetRamp(alpha):
  return ColorMap.setAlpha(ColorMap.HUE_BLUE_TO_RED,rampfloat(0.0,alpha/256,256))
def grayRamp(alpha):
  return ColorMap.setAlpha(ColorMap.GRAY,rampfloat(0.0,alpha/256,256))

def plot(f,g=None,v1=None,v2=None, cmap=None,cmin=None,cmax=None,cint=None,
        label=None,neareast=False,png=None): 
  sp = SimplePlot(SimplePlot.Origin.UPPER_LEFT)
  sp.setVInterval(0.1)
  sp.setHInterval(1.0)
  sp.setVLimits(f1,f1+n1*d1)
  sp.setHLimits(f2,f2+n2*d2)
  sp.setHLabel("Inline (km)")
  sp.setVLabel("Time (s)")
  pxv = sp.addPixels(s1,s2,f);
  pxv.setColorModel(ColorMap.GRAY)
  #pxv.setInterpolation(PixelsView.Interpolation.NEAREST)
  if g:
    pxv.setClips(-1,1)
  else:
    if cmin and cmax:
      pxv.setClips(cmin,cmax)
  if g:
    pv = sp.addPixels(s1,s2,g)
    pv.setInterpolation(PixelsView.Interpolation.NEAREST)
    pv.setColorModel(cmap)
    if cmin and cmax:
      pv.setClips(cmin,cmax)
  cb = sp.addColorBar();
  if cint:
    cb.setInterval(cint)
  if label:
    cb.setLabel(label)
  if (v1 and v2):
    x1 = zerofloat(2)
    x2 = zerofloat(2)
    dx1,dx2 = 6, 14
    scale = 4
    for i2 in range(0,n2,dx2):
      for i1 in range(0,n1,dx1):
        x2[0] = (i2-v2[i2][i1]*scale)*d2+f2
        x2[1] = (i2+v2[i2][i1]*scale)*d2+f2
        x1[0] = (i1-v1[i2][i1]*scale)*d1+f1
        x1[1] = (i1+v1[i2][i1]*scale)*d1+f1
        pvu = sp.addPoints(x1,x2)
        pvu.setLineWidth(4)
        pvu.setLineColor(Color.CYAN)
  sp.plotPanel.setColorBarWidthMinimum(100)
  sp.setSize(1050,700) #for f3d
  sp.setFontSize(30)
  if pngDir and png:
    sp.paintToPng(300,3.333,pngDir+png+".png")

def plot2(f,s1,s2,g=None,cmin=None,cmax=None,label=None,png=None,et=None):
  n2 = len(f)
  n1 = len(f[0])
  panel = panel2Teapot()
  panel = PlotPanel(1,1,PlotPanel.Orientation.X1DOWN_X2RIGHT)
  panel.setHInterval(1.0)
  panel.setVInterval(0.1)
  panel.setHLabel("Inline (km)")
  panel.setVLabel("Time (s)")
  if label:
    panel.addColorBar(label)
  else:
    panel.addColorBar()
  panel.setColorBarWidthMinimum(180)
  pv = panel.addPixels(s1,s2,f)
  pv.setInterpolation(PixelsView.Interpolation.LINEAR)
  pv.setColorModel(ColorMap.GRAY)
  #pv.setClips(min(f),max(f))
  if g:
    alpha = 0.8
  else:
    g = zerofloat(s1.count,s2.count)
    alpha = 0.0
  pv = panel.addPixels(s1,s2,g)
  #pv.setInterpolation(PixelsView.Interpolation.NEAREST)
  #pv.setColorModel(jetFillExceptMin(1.0))
  pv.setColorModel(jetRamp(1.0))
  if cmin and cmax:
    pv.setClips(cmin,cmax)
  frame2Teapot(panel,png)
def panel2Teapot():
  panel = PlotPanel(1,1,
    PlotPanel.Orientation.X1DOWN_X2RIGHT,
    PlotPanel.AxesPlacement.NONE)
  return panel
def frame2Teapot(panel,png=None):
  frame = PlotFrame(panel)
  frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
  #frame.setFontSizeForPrint(8,240)
  #frame.setSize(1240,774)
  frame.setFontSizeForSlide(1.0,0.8)
  frame.setSize(880,700)
  frame.setVisible(True)
  if png and pngDir:
    frame.paintToPng(400,3.2,pngDir+"/"+png+".png")
  return frame
def makePointSets(cmap,f,x1,x2):
  sets = {}
  for i in range(len(f)):
    if f[i] in sets:
      points = sets[f[i]]
      points[0].append(f[i])
      points[1].append(x1[i])
      points[2].append(x2[i])
    else:
      points = [[f[i]],[x1[i]],[x2[i]]] # lists of f, x1, x2
      sets[f[i]] = points
  ns = len(sets)
  fs = zerofloat(1,ns)
  x1s = zerofloat(1,ns)
  x2s = zerofloat(1,ns)
  il = 0
  for points in sets:
    fl = sets[points][0]
    x1l = sets[points][1]
    x2l = sets[points][2]
    nl = len(fl)
    fs[il] = zerofloat(nl)
    x1s[il] = zerofloat(nl)
    x2s[il] = zerofloat(nl)
    copy(fl,fs[il])
    copy(x1l,x1s[il])
    copy(x2l,x2s[il])
    il += 1
  return fs,x1s,x2s

#############################################################################
# utilities

def readImage(name):
  fileName = seismicDir+name+".dat"
  n1,n2 = s1.count,s2.count
  image = zerofloat(n1,n2)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image

def writeImage(name,image):
  fileName = seismicDir+name+".dat"
  aos = ArrayOutputStream(fileName)
  aos.writeFloats(image)
  aos.close()
  return image

#############################################################################
# Run the function main on the Swing thread
import sys
class _RunMain(Runnable):
  def __init__(self,main):
    self.main = main
  def run(self):
    self.main(sys.argv)
def run(main):
  SwingUtilities.invokeLater(_RunMain(main)) 
run(main)
