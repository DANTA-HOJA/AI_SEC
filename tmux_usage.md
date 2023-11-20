# tmux usage

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
