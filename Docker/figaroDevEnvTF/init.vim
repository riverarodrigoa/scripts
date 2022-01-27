set title  " Muestra el nombre del archivo en la ventana de la terminal
set number  " Muestra los números de las líneas
set mouse=a  " Permite la integración del mouse (seleccionar texto, mover el cursor)

set nowrap  " No dividir la línea si es muy larga

set cursorline  " Resalta la línea actual
set colorcolumn=120  " Muestra la columna límite a 120 caracteres

" Indentación a 2 espacios
set tabstop=2
set shiftwidth=2
set softtabstop=2
set shiftround
set expandtab  " Insertar espacios en lugar de <Tab>s
"
set hidden  " Permitir cambiar de buffers sin tener que guardarlos
"
set ignorecase  " Ignorar mayúsculas al hacer una búsqueda
set smartcase  " No ignorar mayúsculas si la palabra a buscar contiene
" mayúsculas
"
set spelllang=en,es  " Corregir palabras usando diccionarios en inglés y
" español
"
set termguicolors  " Activa true colors en la terminal
set background=light  " Fondo del tema: light o dark
colorscheme zellner  " Nombre del tema

let g:mapleader = ' '  " Definir espacio como la tecla líder

nnoremap <leader>s :w<CR>  " Guardar con <líder> + s

nnoremap <leader>e :e $MYVIMRC<CR>  " Abrir el archivo init.vim con <líder> + e
" Usar <líder> + y para copiar al portapapeles
vnoremap <leader>y "+y 
nnoremap <leader>y "+y

" Usar <líder> + d para cortar al portapapeles
vnoremap <leader>d "+d
nnoremap <leader>d "+d

" Usar <líder> + p para pegar desde el portapapeles
nnoremap <leader>p "+p
vnoremap <leader>p "+p
nnoremap <leader>P "+P
vnoremap <leader>P "+P

" Moverse al buffer siguiente con <líder> + l 
nnoremap <leader>l :bnext<CR>
" Moverse al buffer anterior con <líder> + j 
nnoremap <leader>j :bprevious<CR>
" Cerrar el buffer actual con <líder> + q 
nnoremap <leader>q :bdelete<CR>

