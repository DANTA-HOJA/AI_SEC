# Linux Useful Commands / Tools

## General

### tar (pack files)

- Note: 若給定 `abs_path` 打包後的檔案內也會有 `abs_path` 上所有的資料夾

```shell
# copy file ( 特殊用法，功能同 `cp` ，可顯示傳輸速度 )

tar -cf - [/src] | pv | tar -xf - -C [/dst]
```

## SSH

### spacefm (file manager, X11 forwarding)

- 可以開多個 pannel，右鍵選單功能較多

### pcmanfm (file manager, X11 forwarding)

- 照片有縮圖

### geeqie (image viewer, X11 forwarding)

- 不支援 recursive search，但有 GUI 可用

### feh (image viewer, X11 forwarding)

```shell
feh --recursive --auto-zoom --scale-down --geometry 800x600 [/path/to/your/directory]
```

```text
--------------------------------------------------------------------------------
# Command

--auto-zoom   : image < window 會自動放大
--scale-down  : image < window 會自動縮小
--geometry    : 指定初始 window size + 停用切換照片時根據圖片大小重新設定 winodw size

--------------------------------------------------------------------------------
# Control / Keyboard 快捷鍵

d : 在 image 上繪製 image path
r : 重新載入圖片
s : save image in current workingdir, or to the dir if --output-dir is specific
* : 重設 image size ( Zoom to 100% )
/ : 符合是視窗大小

PS. 右鍵可以開啟 menu 有些功能可以從 menu 設定
--------------------------------------------------------------------------------
```

### pqiv (image viewer, X11 forwarding)

- 相較於 feh 有自動刷新資料夾的功能

```shell
pqiv -z 0.15 --watch-directories -P off [/path/to/your/directory]
```

``` text
--------------------------------------------------------------------------------
# Command

-z                  : 控制圖片縮放比例 (1.0 is 100%)
--watch-directories : 監控資料夾，有新檔案自動刷新
-P off              : 取消固定視窗在特定位置

--show-bindings     : 列出快捷鍵

--------------------------------------------------------------------------------
# Control / Keyboard 快捷鍵

前一張影像: backspace
下一張影像: space
--------------------------------------------------------------------------------
```

### tmux (keep termianl section)

- `<prefix>` (default)

    ```shell
    ctrl+b # 先按，不是和其他 cmd 一起按
    ```

- switch to cmd mode

    ```shell
    <prefix> :
    ```

- scroll up and down [[reference]](https://appuals.com/stuck-in-tmux-scroll-up/#:~:text=You%20can%20scroll%20up%20and%20down%20in%20Tmux,page%20down%2C%20etc.%20to%20navigate%20the%20Tmux%20interface.)

    ```shell
    <prefix> [
    ```

- clean history

    ```shell
    <prefix> : clear-history # 清掉已經被吃掉的畫面（run this cmd before new task）
    ```

- save output to file [[reference]](https://unix.stackexchange.com/questions/26548/write-all-tmux-scrollback-to-a-file)

    1. `-S` 用來擷取需要的行數， `-S -` 代表所有的行

        ```shell
        <prefix> : capture-pane -S -
        ```

    2. 儲存檔案

        ```shell
        <prefix> : save-buffer [filename].log
        ```
