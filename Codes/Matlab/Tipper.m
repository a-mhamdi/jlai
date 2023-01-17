function varargout = Tipper(varargin)
% TIPPER MATLAB code for Tipper.fig
%      TIPPER, by itself, creates a new TIPPER or raises the existing
%      singleton*.
%
%      H = TIPPER returns the handle to a new TIPPER or the handle to
%      the existing singleton*.
%
%      TIPPER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TIPPER.M with the given input arguments.
%
%      TIPPER('Property','Value',...) creates a new TIPPER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Tipper_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Tipper_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Tipper

% Last Modified by GUIDE v2.5 22-Nov-2022 20:02:46

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Tipper_OpeningFcn, ...
                   'gui_OutputFcn',  @Tipper_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Tipper is made visible.
function Tipper_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Tipper (see VARARGIN)
fis =readfis('Tipper.fis');
handles.fis = fis;

% Choose default command line output for Tipper
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Tipper wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Tipper_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in compute.
function compute_Callback(hObject, eventdata, handles)
% hObject    handle to compute (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
foodVal = str2num(get(handles.food, 'String'));
serviceVal = str2num(get(handles.service, 'String'));

fis = handles.fis;
tip = evalfis([foodVal, serviceVal], fis);

set(handles.tip, 'String', num2str(tip));


function food_Callback(hObject, eventdata, handles)
% hObject    handle to food (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of food as text
%        str2double(get(hObject,'String')) returns contents of food as a double


% --- Executes during object creation, after setting all properties.
function food_CreateFcn(hObject, eventdata, handles)
% hObject    handle to food (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function service_Callback(hObject, eventdata, handles)
% hObject    handle to service (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of service as text
%        str2double(get(hObject,'String')) returns contents of service as a double


% --- Executes during object creation, after setting all properties.
function service_CreateFcn(hObject, eventdata, handles)
% hObject    handle to service (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tip_Callback(hObject, eventdata, handles)
% hObject    handle to tip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tip as text
%        str2double(get(hObject,'String')) returns contents of tip as a double


% --- Executes during object creation, after setting all properties.
function tip_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
