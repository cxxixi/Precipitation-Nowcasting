from util import *

class ConvLSTMCell(nn.Module):
  """Convolutional LSTM model
  Reference:
    Xingjian Shi et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." 
  """

  def __init__(self, shape, input_channel, filter_size, hidden_size):
    
    super().__init__()
    self._shape = shape
    self._input_channel = input_channel
    self._filter_size = filter_size
    self._hidden_size = hidden_size
    self._conv = nn.Conv2d(in_channels=self._input_channel+self._hidden_size, ###hidden state has similar spational struture as inputs, we simply concatenate them on the feature dimension
                           out_channels=self._hidden_size*4, ##lstm has four gates
                           kernel_size=self._filter_size,
                           stride=1,
                           padding=1)

  def forward(self, x, state):

    _hidden, _cell = state
    # print(x.shape,_hidden.shape)
    cat_x = torch.cat([x, _hidden], dim=1) 
    Conv_x = self._conv(cat_x)
      
    i, f, o, j = torch.chunk(Conv_x, 4, dim = 1)

    i = func.sigmoid(i)
    f = func.sigmoid(f)
    cell = _cell * f + i * func.tanh(j)
    o =func.sigmoid(o)
    hidden = o * func.tanh(cell)

    return hidden, cell